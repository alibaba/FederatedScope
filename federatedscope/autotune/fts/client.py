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
        super(FTSClient,self).__init__(ID, server_id, state, config,
            data, model, device, strategy, is_unseen_client, *args, **kwargs)

        self._diff = config.hpo.fts.diff

        # local file paths
        self.local_bo_path = os.path.join(self._cfg.hpo.working_folder,
            "local_bo_" + str(self.ID) + ".pkl")
        self.local_init_path = os.path.join(self._cfg.hpo.working_folder,
            "local_init_" + str(self.ID) + ".pkl")
        self.local_info_path = os.path.join(self._cfg.hpo.working_folder,
            "local_info_" + str(self.ID) + "_M_" + str(self._cfg.hpo.fts.M) + ".pkl")

        # prepare search space and bounds
        self._ss = parse_search_space(self._cfg.hpo.ss)
        self.dim = len(self._ss)
        self.bounds = np.asarray([(0., 1.) for _ in self._ss])
        self.pbounds = {}
        for k,v in self._ss.items():
            if not (hasattr(v, 'lower') and  hasattr(v, 'upper')):
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

    def _obj_func(self, x):
        """
        Run local evaluation, return some metric to maximize (e.g. val_acc)
        """
        hyperparams = x2conf(x, self.pbounds, self._ss)
        self._apply_hyperparams(hyperparams)
        sample_size, model_para_all, results = self.trainer.train()
        if self._diff:
            res = results['val_avg_loss_before'] - results['val_avg_loss_after']
        else:
            res = - results['val_avg_loss_after']
        return res

    def _generate_agent_info(self, rand_feats):
        logger.info(('-'*20, ' generate info on clinet %d ' % self.ID , '_'*20))
        v_kernel = self._cfg.hpo.fts.v_kernel
        obs_noise = self._cfg.hpo.fts.obs_noise
        M = self._cfg.hpo.fts.M
        tns = self._cfg.hpo.fts.tns

        # run standard BO locally
        max_iter = self._cfg.hpo.fts.local_bo_max_iter
        gp_opt_schedule = self._cfg.hpo.fts.gp_opt_schedule
        pt = np.ones(max_iter + 5)
        LocalBO(
            cid = self.ID,
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
            M_target=200
        ).maximize(n_iter=max_iter, init_points=3)

        # generate local RFF information
        res = pickle.load(open(self.local_bo_path, "rb"))
        ys = np.array(res["all"]["values"]).reshape(-1, 1)
        params = np.array(res["all"]["params"])
        xs = np.array(params)
        xs, ys = xs[:tns], ys[:tns]
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
        w_samples = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
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
            curr_y = self._obj_func(x_max)
            content = {
                'is_required_agent_info': False,
                'curr_y': curr_y,
            }
            hyper_param = x2conf(x_max, self.pbounds, self._ss)
            logger.info('{Client: %d, ' % self.ID + \
                        'GP_opt_iter: %d, ' % round + \
                        'Params: ' + str(hyper_param) + \
                        ', Perform: ' + str(curr_y) + \
                        '}'
                        )

        self.state = round
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=content))