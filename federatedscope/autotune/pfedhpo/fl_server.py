import copy
import json
import os
import logging
from itertools import product
import pickle

import torch.nn
import yaml

import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
from torch.nn import functional as F

from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.autotune.pfedhpo.utils import *
from federatedscope.autotune.utils import parse_search_space
from federatedscope.core.auxiliaries.utils import logfile_2_wandb_dict, logline_2_wandb_dict
from federatedscope.core.auxiliaries.utils import merge_dict, merge_param_dict

logger = logging.getLogger(__name__)


class pFedHPOFLServer(Server):
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

        # initialize action space and the policy
        with open(config.hpo.pfedhpo.ss, 'r') as ips:
            ss = yaml.load(ips, Loader=yaml.FullLoader)

        if next(iter(ss.keys())).startswith('arm'):
            # This is a flattened action space
            # ensure the order is unchanged
            ss = sorted([(int(k[3:]), v) for k, v in ss.items()],
                        key=lambda x: x[0])
            self._grid = []
            self._cfsp = [[tp[1] for tp in ss]]
        else:
            # This is not a flat search space
            # be careful for the order
            self._grid = sorted(ss.keys())
            self._cfsp = [ss[pn] for pn in self._grid]

        super(pFedHPOFLServer, self).__init__(
            ID, state, config, data, model, client_num,
            total_round_num, device, strategy, **kwargs)
        os.makedirs(self._cfg.hpo.working_folder, exist_ok=True)

        if self._cfg.federate.restore_from != '':
            if not os.path.exists(self._cfg.federate.restore_from):
                logger.warning(f'Invalid `restore_from`:'
                               f' {self._cfg.federate.restore_from}.')
            else:
                pi_ckpt_path = self._cfg.federate.restore_from[
                               :self._cfg.federate.restore_from.rfind('.')] \
                               + "_pfedhpo.yaml"
                with open(pi_ckpt_path, 'r') as ips:
                    ckpt = yaml.load(ips, Loader=yaml.FullLoader)
                self._z = [np.asarray(z) for z in ckpt['z']]
                self._theta = [np.exp(z) for z in self._z]
                self._store = ckpt['store']
                self._stop_exploration = ckpt['stop']
                self._trace = dict()
                self._trace['global'] = ckpt['global']
                self._trace['refine'] = ckpt['refine']
                self._trace['entropy'] = ckpt['entropy']
                self._trace['mle'] = ckpt['mle']

        self.train_anchor = self._cfg.hpo.pfedhpo.train_anchor

        # prepare search space and bounds
        self._ss = parse_search_space(self._cfg.hpo.pfedhpo.ss)
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

        # prepare hyper-net
        self.client2idx = None

        if not self.train_anchor:
            hyper_enc = torch.load(os.path.join(self._cfg.hpo.working_folder,
                                                'hyperNet_encoding.pt'))
            self.client_encoding = hyper_enc['encoding'].to(self._cfg.device)
            self.HyperNet = HyperNet(input_dim=self.client_encoding.shape[1],
                                     num_params=len(self.pbounds),
                                     n_clients=client_num).to(self._cfg.device)
            self.HyperNet.load_state_dict(hyper_enc['hyperNet'])
            self.raw_params = self.HyperNet(self.client_encoding)[0].detach().cpu().numpy()


    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = content

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(tuple(content[0:2]))

        return self.check_and_move_on()

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast
                behavior. And sample_client_num = -1 denotes to broadcast to
                all the clients.
            filter_unseen_clients: whether filter out the unseen clients that
                do not contribute to FL process by training on their local
                data and uploading their local model update. The splitting is
                useful to check participation generalization gap in [ICLR'22,
                What Do We Mean by Generalization in Federated Learning?]
                You may want to set it to be False when in evaluation stage
        """

        if self.train_anchor:
            ckpt_path = os.path.join(self._cfg.hpo.working_folder,
                                     'temp_model_round_%d.pt' % (int(self.state)))
            torch.save(self.model.state_dict(), ckpt_path)

        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.model.state_dict()

        if not self.client2idx:
            client2idx = {}
            _all_clients = list(self.comm_manager.neighbors.keys())
            for i, k in zip(range(len(_all_clients)), _all_clients):
                client2idx[k] = i
            self.client2idx = client2idx

        for rcv_idx in receiver:
            if self.train_anchor:
                sampled_cfg = None
            else:
                sampled_cfg = x2conf(self.raw_params[self.client2idx[rcv_idx]], self.pbounds, self._ss)

            content = {
                'model_param': model_para,
                'hyper_param': sampled_cfg
            }
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))

        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def save_res(self, feedbacks):
        feedbacks = {'round': self.state,
                     'results': feedbacks}
        line = str(feedbacks) + "\n"
        with open(os.path.join(self._cfg.hpo.working_folder,
                               'anchor_eval_results.log'), "a") as outfile:
            outfile.write(line)

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer, when enough messages are receiving,
        trigger some events (such as perform aggregation, evaluation,
        and move to the next training round)
        """
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result:
            min_received_num = len(list(self.comm_manager.neighbors.keys()))

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):

            if not check_eval_result:  # in the training process
                mab_feedbacks = dict()
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for model_idx in range(self.model_num):
                    model = self.models[model_idx]
                    aggregator = self.aggregators[model_idx]
                    msg_list = list()
                    for client_id in train_msg_buffer:
                        if self.model_num == 1:
                            msg_list.append(
                                tuple(train_msg_buffer[client_id][0:2]))
                        else:
                            train_data_size, model_para_multiple = \
                                train_msg_buffer[client_id][0:2]
                            msg_list.append((train_data_size,
                                             model_para_multiple[model_idx]))

                        # collect feedbacks for updating the policy
                        if model_idx == 0:
                            mab_feedbacks[client_id] = train_msg_buffer[client_id][2]

                    # Trigger the monitor here (for training)
                    if 'dissim' in self._cfg.eval.monitoring:
                        from federatedscope.core.auxiliaries.utils import \
                            calc_blocal_dissim
                        # TODO: fix load_state_dict
                        B_val = calc_blocal_dissim(
                            model.load_state_dict(strict=False), msg_list)
                        formatted_eval_res = self._monitor.format_eval_res(
                            B_val, rnd=self.state, role='Server #')
                        logger.info(formatted_eval_res)

                    # Aggregate
                    agg_info = {
                        'client_feedback': msg_list,
                        'recover_fun': self.recover_fun
                    }
                    result = aggregator.aggregate(agg_info)
                    model.load_state_dict(result, strict=False)

                self.state += 1
                #  Evaluate
                logger.info(
                    'Server: Starting evaluation at round {:d}.'.format(
                        self.state))
                self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                logger.info('-'*30)
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()

            if self.train_anchor and self.history_results:
                with open(os.path.join(self._cfg.hpo.working_folder,
                                       'anchor_eval_results.json'), 'w') as f:
                    json.dump(self.history_results, f)
        else:
            move_on_flag = False

        return move_on_flag
