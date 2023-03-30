import os
import logging
import json
import copy
import pickle
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.workers import Client

from federatedscope.autotune.fts.utils import *
from federatedscope.autotune.utils import parse_search_space
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

logger = logging.getLogger(__name__)


class FTSClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FTSClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, is_unseen_client, *args, **kwargs)
        self.data = data
        self.model = model
        self.device = device
        self._diff = config.hpo.fts.diff
        self._init_model = copy.deepcopy(model)

        # local file paths
        self.local_bo_path = os.path.join(self._cfg.hpo.working_folder,
                                          "local_bo_" + str(self.ID) + ".pkl")
        self.local_init_path = os.path.join(
            self._cfg.hpo.working_folder,
            "local_init_" + str(self.ID) + ".pkl")
        self.local_info_path = os.path.join(
            self._cfg.hpo.working_folder, "local_info_" + str(self.ID) +
            "_M_" + str(self._cfg.hpo.fts.M) + ".pkl")

        # prepare search space and bounds
        self._ss = parse_search_space(self._cfg.hpo.fts.ss)
        self.dim = len(self._ss)
        self.bounds = np.asarray([(0., 1.) for _ in self._ss])
        self.pbounds = {}
        for k, v in self._ss.items():
            if not (hasattr(v, 'lower') and hasattr(v, 'upper')):
                raise ValueError("Unsupported hyper type {}".format(type(v)))
            else:
                if v.log:
                    l, u = np.log10(v.lower), np.log10(v.upper)
                else:
                    l, u = v.lower, v.upper
                self.pbounds[k] = (l, u)

    def _apply_hyperparams(self, hyperparams):
        """Apply the given hyperparameters
        Arguments:
            hyperparams (dict): keys are hyperparameter names \
                and values are specific choices.
        """

        cmd_args = []
        for k, v in hyperparams.items():
            cmd_args.append(k)
            cmd_args.append(v)

        self._cfg.defrost()
        self._cfg.merge_from_list(cmd_args)
        self._cfg.freeze(inform=False)

        self.trainer.ctx.setup_vars()

    def _get_new_trainer(self):
        self.model = copy.deepcopy(self._init_model)
        self.trainer = get_trainer(model=self.model,
                                   data=self.data,
                                   device=self.device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)

    def _obj_func(self, x, return_eval=False):
        """
        Run local evaluation, return some metric to maximize (e.g. val_acc)
        """
        self._get_new_trainer()

        baseline = 5.0
        hyperparams = x2conf(x, self.pbounds, self._ss)
        self._apply_hyperparams(hyperparams)

        results_before = self.trainer.evaluate('val')
        for _ in range(self._cfg.hpo.fts.local_bo_epochs):
            sample_size, model_para_all, results = self.trainer.train()
        results_after = self.trainer.evaluate('val')

        if self._diff:
            res = results_before['val_avg_loss'] \
                  - results_after['val_avg_loss']
        else:
            res = baseline - results_after['val_avg_loss']
        if return_eval:
            return res, results_after
        else:
            return res

    def _generate_agent_info(self, rand_feats):
        logger.info(
            ('-' * 20, ' generate info on clinet %d ' % self.ID, '_' * 20))
        v_kernel = self._cfg.hpo.fts.v_kernel
        obs_noise = self._cfg.hpo.fts.obs_noise
        M = self._cfg.hpo.fts.M
        M_target = self._cfg.hpo.fts.M_target

        # run standard BO locally
        max_iter = self._cfg.hpo.fts.local_bo_max_iter
        gp_opt_schedule = self._cfg.hpo.fts.gp_opt_schedule
        pt = np.ones(max_iter + 5)
        LocalBO(cid=self.ID,
                f=self._obj_func,
                bounds=self.bounds,
                keys=list(self.pbounds.keys()),
                gp_opt_schedule=gp_opt_schedule,
                use_init=None,
                log_file=self.local_bo_path,
                save_init=True,
                save_init_file=self.local_init_path,
                pt=pt,
                P_N=None,
                ls=self._cfg.hpo.fts.ls,
                var=self._cfg.hpo.fts.var,
                g_var=self._cfg.hpo.fts.g_var,
                N=self._cfg.federate.client_num - 1,
                M_target=M_target).maximize(n_iter=max_iter, init_points=3)

        # generate local RFF information
        res = pickle.load(open(self.local_bo_path, "rb"))
        ys = np.array(res["all"]["values"]).reshape(-1, 1)
        params = np.array(res["all"]["params"])
        xs = np.array(params)
        xs, ys = xs[:max_iter], ys[:max_iter]
        Phi = np.zeros((xs.shape[0], M))

        s, b = rand_feats["s"], rand_feats["b"]
        for i, x in enumerate(xs):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features
            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), ys)
        w_samples = np.random.multivariate_normal(np.squeeze(nu_t),
                                                  obs_noise * Sigma_t_inv, 1)
        pickle.dump(w_samples, open(self.local_info_path, "wb"))

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        require_agent_infos = content['require_agent_infos']

        # generate local info and init then send them to server
        if require_agent_infos:
            rand_feat = content['random_feats']
            self._generate_agent_info(rand_feat)
            agent_info = pickle.load(open(self.local_info_path, "rb"))
            agent_init = pickle.load(open(self.local_init_path, "rb"))
            content = {
                'is_required_agent_info': True,
                'agent_info': agent_info,
                'agent_init': agent_init,
            }

        # local run on given hyper-param and return performance
        else:
            x_max = content['x_max']
            curr_y, eval_res = self._obj_func(x_max, return_eval=True)
            content = {
                'is_required_agent_info': False,
                'curr_y': curr_y,
            }
            hyper_param = x2conf(x_max, self.pbounds, self._ss)
            logger.info('{Client: %d, ' % self.ID +
                        'GP_opt_iter: %d, ' % round + 'Params: ' +
                        str(hyper_param) + ', Perform: ' + str(curr_y) + '}')
            logger.info(
                self._monitor.format_eval_res(eval_res,
                                              rnd=self.state,
                                              role='Client #{}'.format(
                                                  self.ID),
                                              return_raw=True))

        self.state = round
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=content))

    def callback_funcs_for_evaluate(self, message: Message):
        round, sender, content = \
            message.state, message.sender, message.content
        require_agent_infos = content['require_agent_infos']
        assert not require_agent_infos, \
            "Can not evaluate when there is no agents' information"

        self.state = message.state
        self._obj_func(content['x_max'])

        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(target_data_split_name=split)
            for key in eval_metrics:
                if self._cfg.federate.mode == 'distributed':
                    logger.info('Client #{:d}: (Evaluation ({:s} set) at '
                                'Round #{:d}) {:s} is {:.6f}'.format(
                                    self.ID, split, self.state, key,
                                    eval_metrics[key]))
                metrics.update(**eval_metrics)

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
