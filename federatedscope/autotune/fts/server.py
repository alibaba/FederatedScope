import copy
import os
import logging
from itertools import product
import pickle
import yaml

import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
import GPy

from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.autotune.fts.utils import *
from federatedscope.autotune.utils import parse_search_space

logger = logging.getLogger(__name__)


class FTSServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(FTSServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        assert self.sample_client_num == self._cfg.federate.client_num

        self.util_ts = UtilityFunction(kind="ts")
        self.M = self._cfg.hpo.fts.M

        # server file paths
        self.all_lcoal_init_path = os.path.join(self._cfg.hpo.working_folder,
                                                "all_localBO_inits.pkl")
        self.all_local_info_path = os.path.join(self._cfg.hpo.working_folder,
                                                "all_localBO_infos.pkl")
        self.rand_feat_path = os.path.join(
            self._cfg.hpo.working_folder,
            "rand_feat_M_" + str(self.M) + ".pkl")

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

        # distribution used to sample GP models
        pt = 1 - 1 / (np.arange(self._cfg.hpo.fts.fed_bo_max_iter + 5) +
                      1)**2.0
        pt[0] = pt[1]
        self.pt = pt
        self.num_other_clients = self._cfg.federate.client_num - 1
        self.ws = np.ones(self.num_other_clients) / self.num_other_clients

        # records for all GP models
        N = self._cfg.federate.client_num + 1
        self.x_max = [None for _ in range(N)]
        self.y_max = [None for _ in range(N)]
        self.X = [None for _ in range(N)]
        self.Y = [None for _ in range(N)]
        self.incumbent = [None for _ in range(N)]
        self.gp = [None for _ in range(N)]
        self.gp_params = [None for _ in range(N)]
        self.initialized = [False for _ in range(N)]
        self.res = [{
            'max_value': None,
            'max_param': None,
            'all_values': [],
            'all_params': [],
        } for _ in range(N)]
        self.res_paths = [
            os.path.join(self._cfg.hpo.working_folder,
                         "result_params_%d.pkl" % cid) for cid in range(N)
        ]

        # load or generate agent_info, agent_init, and rand_feat.
        # if files already exit, load from saved files;
        # else require the clients in the first round
        if os.path.exists(self.all_local_info_path) and \
                self._cfg.hpo.fts.allow_load_existing_info:
            logger.info('Using existing rand_feat, agent_infos, '
                        'and agent_inits')
            self.require_agent_infos = False
            self.state = 1
            self.random_feats = pickle.load(open(self.rand_feat_path, 'rb'))
            self.all_agent_info = pickle.load(
                open(self.all_local_info_path, 'rb'))
            self.all_agent_init = pickle.load(
                open(self.all_lcoal_init_path, 'rb'))
        else:
            self.require_agent_infos = True
            self.random_feats = self._generate_shared_rand_feats()
            self.all_agent_info = {}
            self.all_agent_init = {}

        # point out target clients need to be optimized by FTS
        if not config.hpo.fts.target_clients:
            self.target_clients = list(
                range(1, self._cfg.federate.client_num + 1))
        else:
            self.target_clients = list(config.hpo.fts.target_clients)

    def _generate_shared_rand_feats(self):
        # generate shared random features
        ls = self._cfg.hpo.fts.ls
        v_kernel = self._cfg.hpo.fts.v_kernel
        obs_noise = self._cfg.hpo.fts.obs_noise
        M = self.M

        s = np.random.multivariate_normal(np.zeros(self.dim),
                                          1 / ls**2 * np.identity(self.dim), M)
        b = np.random.uniform(0, 2 * np.pi, M)
        random_features = {
            "M": M,
            "length_scale": ls,
            "s": s,
            "b": b,
            "obs_noise": obs_noise,
            "v_kernel": v_kernel
        }
        pickle.dump(random_features, open(self.rand_feat_path, "wb"))

        return random_features

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients
        """
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if self.require_agent_infos:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')
        else:
            receiver = list(self.target_clients)

        for rcv_idx in receiver:
            if self.require_agent_infos:
                content = {
                    'require_agent_infos': True,
                    'random_feats': self.random_feats,
                }

            else:
                # initialize gp models and init points
                if not self.initialized[rcv_idx]:
                    init = self.all_agent_init[rcv_idx]
                    self.X[rcv_idx] = init['X']
                    self.Y[rcv_idx] = init['Y']
                    self.incumbent[rcv_idx] = np.max(self.Y[rcv_idx])
                    logger.info("Using pre-existing initializations "
                                "for client {} with {} points".format(
                                    rcv_idx, len(self.Y[rcv_idx])))

                    y_max = np.max(self.Y[rcv_idx])
                    self.y_max[rcv_idx] = y_max

                    ur = unique_rows(self.X[rcv_idx])
                    self.gp[rcv_idx] = GPy.models.GPRegression(
                        self.X[rcv_idx][ur],
                        self.Y[rcv_idx][ur].reshape(-1, 1),
                        GPy.kern.RBF(input_dim=self.X[rcv_idx].shape[1],
                                     lengthscale=self._cfg.hpo.fts.ls,
                                     variance=self._cfg.hpo.fts.var,
                                     ARD=False))
                    self.gp[rcv_idx]["Gaussian_noise.variance"][0] = \
                        self._cfg.hpo.fts.g_var
                    self._opt_gp(rcv_idx)
                    self.initialized[rcv_idx] = True

                # sample hyper from this client's GP or others' GP
                info_ts = copy.deepcopy(self.all_agent_info)
                del info_ts[rcv_idx]
                info_ts = list(info_ts.values())

                def _try_sample_try(func):
                    _loop = True
                    while _loop:
                        try:
                            x_max, all_ucb = func(rcv_idx, self.y_max[rcv_idx],
                                                  self.state, info_ts)
                            _loop = False
                        except:
                            _loop = True
                    return x_max, all_ucb

                if np.random.random() < self.pt[self.state - 1]:
                    x_max, all_ucb = _try_sample_try(self._sample_from_this)
                else:
                    x_max, all_ucb = _try_sample_try(self._sample_from_others)
                self.x_max[rcv_idx] = x_max

                content = {
                    'require_agent_infos': False,
                    'x_max': x_max,
                }

            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def _sample_from_this(self, client, y_max, iteration, info_ts):
        M_target = self._cfg.hpo.fts.M_target

        ls_target = self.gp[client]["rbf.lengthscale"][0]
        v_kernel = self.gp[client]["rbf.variance"][0]
        obs_noise = self.gp[client]["Gaussian_noise.variance"][0]

        s = np.random.multivariate_normal(
            np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim),
            M_target)
        b = np.random.uniform(0, 2 * np.pi, M_target)
        random_features_target = {
            "M": M_target,
            "length_scale": ls_target,
            "s": s,
            "b": b,
            "obs_noise": obs_noise,
            "v_kernel": v_kernel
        }

        Phi = np.zeros((self.X[client].shape[0], M_target))
        for i, x in enumerate(self.X[client]):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(
                2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features
            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T),
                      self.Y[client].reshape(-1, 1))
        w_sample = np.random.multivariate_normal(np.squeeze(nu_t),
                                                 obs_noise * Sigma_t_inv, 1)

        x_max, all_ucb = acq_max(ac=self.util_ts.utility,
                                 gp=self.gp[client],
                                 M=M_target,
                                 N=self.num_other_clients,
                                 gp_samples=None,
                                 random_features=random_features_target,
                                 info_ts=info_ts,
                                 pt=self.pt,
                                 ws=self.ws,
                                 use_target_label=True,
                                 w_sample=w_sample,
                                 y_max=y_max,
                                 bounds=self.bounds,
                                 iteration=iteration)
        return x_max, all_ucb

    def _sample_from_others(self, client, y_max, iteration, info_ts):
        agent_ind = np.arange(self.num_other_clients)
        random_agent_n = np.random.choice(agent_ind, 1, p=self.ws)[0]
        w_sample = info_ts[random_agent_n]
        x_max, all_ucb = acq_max(ac=self.util_ts.utility,
                                 gp=self.gp[client],
                                 M=self.M,
                                 N=self.num_other_clients,
                                 gp_samples=None,
                                 random_features=self.random_feats,
                                 info_ts=info_ts,
                                 pt=self.pt,
                                 ws=self.ws,
                                 use_target_label=False,
                                 w_sample=w_sample,
                                 y_max=y_max,
                                 bounds=self.bounds,
                                 iteration=iteration)
        return x_max, all_ucb

    def _opt_gp(self, client):
        self.gp[client].optimize_restarts(num_restarts=10,
                                          messages=False,
                                          verbose=False)
        self.gp_params[client] = self.gp[client].parameters
        # print("---Optimized Hyper of Client %d : " % client, self.gp[client])

    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = content

        return self.check_and_move_on()

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving,
        some events (such as perform aggregation, evaluation, and move to
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for
            evaluation; and check the message buffer for training otherwise.
        """
        if min_received_num is None or check_eval_result:
            min_received_num = len(self.target_clients)
        if self.require_agent_infos:
            min_received_num = self._cfg.federate.client_num

        move_on_flag = True

        # When enough messages are receiving
        if self.check_buffer(self.state, min_received_num, check_eval_result):

            if not check_eval_result:
                # The first round is to collect clients' infomation,
                # receive agent_info
                if self.require_agent_infos:
                    for _client, _content in \
                            self.msg_buffer['train'][self.state].items():
                        assert _content['is_required_agent_info']
                        self.all_agent_info[_client] = _content['agent_info']
                        self.all_agent_init[_client] = _content['agent_init']
                    pickle.dump(self.all_agent_info,
                                open(self.all_local_info_path, "wb"))
                    pickle.dump(self.all_agent_init,
                                open(self.all_lcoal_init_path, "wb"))
                    self.require_agent_infos = False

                # Other rounds are to update GP models, receive performance
                else:
                    for _client, _content in \
                            self.msg_buffer['train'][self.state].items():
                        curr_y = _content['curr_y']
                        self.Y[_client] = np.append(self.Y[_client], curr_y)
                        self.X[_client] = np.vstack(
                            (self.X[_client], self.x_max[_client].reshape(
                                (1, -1))))
                        if self.Y[_client][-1] > self.y_max[_client]:
                            self.y_max[_client] = self.Y[_client][-1]
                            self.incumbent[_client] = self.Y[_client][-1]
                        ur = unique_rows(self.X[_client])
                        self.gp[_client].set_XY(X=self.X[_client][ur],
                                                Y=self.Y[_client][ur].reshape(
                                                    -1, 1))

                        _schedule = self._cfg.hpo.fts.gp_opt_schedule
                        if self.state >= _schedule \
                                and self.state % _schedule == 0:
                            self._opt_gp(_client)

                        x_max_param = self.X[_client][self.Y[_client].argmax()]
                        hyper_param = x2conf(x_max_param, self.pbounds,
                                             self._ss)
                        self.res[_client]['max_param'] = hyper_param
                        self.res[_client]['max_value'] = self.Y[_client].max()
                        self.res[_client]['all_values'].append(
                            self.Y[_client][-1].tolist())
                        self.res[_client]['all_params'].append(
                            self.X[_client][-1].tolist())
                        pickle.dump(self.res[_client],
                                    open(self.res_paths[_client], 'wb'))

                self.state += 1
                if self.state <= self._cfg.hpo.fts.fed_bo_max_iter:
                    # Move to next round of training
                    logger.info(f'----------- GP optimizing iteration (Round '
                                f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.broadcast_model_para()
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                _results = {}
                for _client, _content in \
                        self.msg_buffer['eval'][self.state].items():
                    _results[_client] = _content

                self.history_results = merge_dict(self.history_results,
                                                  _results)
                if self.state > self._cfg.hpo.fts.fed_bo_max_iter:
                    self.check_and_save()
        else:
            move_on_flag = False

        return move_on_flag

    def check_and_save(self):
        """
        To save the results and save model after each evaluation
        """
        # early stopping
        should_stop = False

        formatted_best_res = self._monitor.format_eval_res(
            results=self.history_results,
            rnd="Final",
            role='Server #',
            forms=["raw"],
            return_raw=True)
        logger.info('*' * 50)
        logger.info(formatted_best_res)
        self._monitor.save_formatted_results(formatted_best_res)

        if should_stop:
            self.state = self.total_round_num + 1

        logger.info('*' * 50)
        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1
