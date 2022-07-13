import torch
import logging
import copy
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.worker.server import Server
from federatedscope.core.worker.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.gfl.gcflplus.utils import compute_pairwise_distances, \
    min_cut, norm

logger = logging.getLogger(__name__)


class GCFLPlusServer(Server):
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
        super(GCFLPlusServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        # Initial cluster
        self.cluster_indices = [
            np.arange(1, self._cfg.federate.client_num + 1).astype("int")
        ]
        self.client_clusters = [[ID for ID in cluster_id]
                                for cluster_id in self.cluster_indices]
        # Maintain a grad sequence
        self.seqs_grads = {
            idx: []
            for idx in range(1, self._cfg.federate.client_num + 1)
        }

    def compute_update_norm(self, cluster):
        max_norm = -np.inf
        cluster_dWs = []
        for key in cluster:
            content = self.msg_buffer['train'][self.state][key]
            _, model_para, client_dw, _ = content
            dW = {}
            for k in model_para.keys():
                dW[k] = client_dw[k]
            update_norm = norm(dW)
            if update_norm > max_norm:
                max_norm = update_norm
            cluster_dWs.append(
                torch.cat([value.flatten() for value in dW.values()]))
        mean_norm = torch.norm(torch.mean(torch.stack(cluster_dWs),
                                          dim=0)).item()
        return max_norm, mean_norm

    def check_and_move_on(self, check_eval_result=False):

        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):

            if not check_eval_result:  # in the training process
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for model_idx in range(self.model_num):
                    model = self.models[model_idx]
                    aggregator = self.aggregators[model_idx]
                    msg_list = list()
                    for client_id in train_msg_buffer:
                        if self.model_num == 1:
                            train_data_size, model_para, _, convGradsNorm = \
                                train_msg_buffer[client_id]
                            self.seqs_grads[client_id].append(convGradsNorm)
                            msg_list.append((train_data_size, model_para))
                        else:
                            raise ValueError(
                                'GCFL server not support multi-model.')

                    cluster_indices_new = []
                    for cluster in self.cluster_indices:
                        max_norm, mean_norm = self.compute_update_norm(cluster)
                        # create new cluster
                        if mean_norm < self._cfg.gcflplus.EPS_1 and max_norm\
                                > self._cfg.gcflplus.EPS_2 and len(
                                cluster) > 2 and self.state > 20 and all(
                                    len(value) >= self._cfg.gcflplus.seq_length
                                    for value in self.seqs_grads.values()):
                            _, model_para_cluster, _, _ = self.msg_buffer[
                                'train'][self.state][cluster[0]]
                            tmp = [
                                self.seqs_grads[ID]
                                [-self._cfg.gcflplus.seq_length:]
                                for ID in cluster
                            ]
                            dtw_distances = compute_pairwise_distances(
                                tmp, self._cfg.gcflplus.standardize)
                            c1, c2 = min_cut(
                                np.max(dtw_distances) - dtw_distances, cluster)
                            cluster_indices_new += [c1, c2]
                            # reset seqs_grads for all clients
                            self.seqs_grads = {
                                idx: []
                                for idx in range(
                                    1, self._cfg.federate.client_num + 1)
                            }
                        # keep this cluster
                        else:
                            cluster_indices_new += [cluster]

                    self.cluster_indices = cluster_indices_new
                    self.client_clusters = [[
                        ID for ID in cluster_id
                    ] for cluster_id in self.cluster_indices]

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at round {:d}.'.
                        format(self.ID, self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    for cluster in self.cluster_indices:
                        msg_lsit = list()
                        for key in cluster:
                            content = self.msg_buffer['train'][self.state -
                                                               1][key]
                            train_data_size, model_para, client_dw,  \
                                convGradsNorm = content
                            msg_lsit.append((train_data_size, model_para))

                        agg_info = {
                            'client_feedback': msg_list,
                            'recover_fun': self.recover_fun
                        }
                        result = aggregator.aggregate(agg_info)
                        model.load_state_dict(result, strict=False)
                        # aggregator.update(result)
                        # Send to Clients
                        self.comm_manager.send(
                            Message(msg_type='model_para',
                                    sender=self.ID,
                                    receiver=cluster.tolist(),
                                    state=self.state,
                                    content=result))

                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new traininground(Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                else:
                    # Final Evaluate
                    logger.info('Server #{:d}: Training is finished! Starting '
                                'evaluation.'.format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()


class GCFLPlusClient(Client):
    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # Cache old W
        W_old = copy.deepcopy(content)
        self.trainer.update(content)
        self.state = round
        sample_size, model_para, results = self.trainer.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para = copy.deepcopy(model_para)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID)))

        # Compute norm of W & norm of grad
        dW = dict()
        for key in model_para.keys():
            dW[key] = model_para[key] - W_old[key].cpu()

        self.W = {key: value for key, value in self.model.named_parameters()}

        convGradsNorm = dict()
        for key in model_para.keys():
            if key in self.W and self.W[key].grad is not None:
                convGradsNorm[key] = self.W[key].grad
        convGradsNorm = norm(convGradsNorm)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para, dW, convGradsNorm)))
