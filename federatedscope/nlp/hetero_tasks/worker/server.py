import os
import json
import logging
import copy
import torch
import numpy as np
from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.nlp.hetero_tasks.trainer.utils import ContrastiveMonitor
from federatedscope.nlp.hetero_tasks.dataset.utils import load_synth_data

logger = logging.getLogger(__name__)


class ATCServer(Server):
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
                 unseen_clients_id=None,
                 **kwargs):

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         unseen_clients_id=unseen_clients_id,
                         **kwargs)

        # multiple models are maintained for different clients
        self.models = [
            copy.deepcopy(self.model) for _ in range(self.client_num)
        ]
        self.tasks = [
            config.model.pretrain_tasks[0]
            if config.model.pretrain_tasks else None
            for _ in range(self.client_num)
        ]
        self.atc_vanilla = config.federate.atc_vanilla
        if not self.atc_vanilla:
            self.aggregator.update_models(self.models)
            self.aggregator.update_neighbors(self.comm_manager.neighbors)

        self.use_contrastive_loss = self._cfg.model.use_contrastive_loss
        if self._cfg.model.stage == 'contrast':
            # load synthetic for contrastive learning
            synth_feats, synth_toks = load_synth_data(self._cfg.data)
            self.contrast_monitor = ContrastiveMonitor()
            self.contrast_monitor.update_enc_hidden(synth_feats)
            self.contrast_monitor.update_synth_tokens(synth_toks)
            self.aggregator.update_contrast_monitor(self.contrast_monitor)

    def _perform_federated_aggregation(self):
        train_msg_buffer = dict(
            sorted(self.msg_buffer['train'][self.state].items(),
                   key=lambda x: x[0]))
        msg_list = list()
        for client_id in train_msg_buffer:
            msg_list.append(train_msg_buffer[client_id])

        # Aggregate
        aggregated_num = len(msg_list)
        if self.atc_vanilla:
            agg_info = {
                'client_feedback': [[x['sample_size'], x['model_para']]
                                    for x in msg_list],
                'recover_fun': self.recover_fun,
            }
            avg_models = self.aggregator.aggregate(agg_info)
            tasks = [None for _ in range(self.client_num)]
            for i in range(self.client_num):
                self.models[i].load_state_dict(avg_models, strict=False)
        else:
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }
            avg_models, tasks = self.aggregator.aggregate(agg_info)
            if avg_models is not None and 'model_para' in avg_models:
                for i in range(self.client_num):
                    self.models[i].load_state_dict(avg_models['model_para'][i],
                                                   strict=False)
        self.tasks = tasks

        if self.use_contrastive_loss:
            if self._cfg.model.task != 'pretrain' and \
                    self.contrast_monitor.stat == 2:
                self.msg_buffer['train'][self.state].clear()
                self.broadcast_model_para(
                    msg_type='model_para',
                    sample_client_num=self.sample_client_num)
                return -1
            if self.contrast_monitor.stat == 3:
                self.contrast_monitor.reset()

        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            sample_ids = np.random.choice(np.arange(self.client_num),
                                          size=sample_client_num,
                                          replace=False).tolist()
        else:
            sample_ids = list(range(self.client_num))

        receivers = sorted(list(self.comm_manager.neighbors.keys()))
        model_para = [model.state_dict() for model in self.models]
        skip_broadcast = self._cfg.federate.method in ['local', 'global']
        if skip_broadcast:
            model_para = [{} for _ in self.models]

        for i in sample_ids:
            if not self.use_contrastive_loss:
                content = {
                    'model_para': model_para[i],
                    'task': self.tasks[i],
                }
            else:
                content = {
                    'model_para': model_para[i],
                    'task': self.tasks[i],
                    'contrast_monitor': self.contrast_monitor,
                }
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receivers[i],
                        state=self.state,
                        content=content))

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def merge_eval_results_from_all_clients(self, final_round=False):
        state = self.state if not final_round else self.state - 1
        eval_msg_buffer = self.msg_buffer['eval'][state]

        if 'group_avg' in self._cfg.eval.report:
            metrics_all_clients = eval_msg_buffer
        else:
            metrics_all_clients = dict()
            for each_client in eval_msg_buffer:
                client_eval_results = eval_msg_buffer[each_client]
                for key in client_eval_results.keys():
                    res = client_eval_results[key]
                    if isinstance(res, dict):
                        for k, v in res.items():
                            cur_key = key + '_' + k
                            if key not in metrics_all_clients:
                                metrics_all_clients[cur_key] = list()
                            metrics_all_clients[cur_key].append(float(v))
                    else:
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(float(res))
        formatted_logs = self._monitor.format_eval_res(
            metrics_all_clients,
            rnd=self.state + 1,
            role='Server #',
            forms=self._cfg.eval.report)
        logger.info(formatted_logs)
        self._monitor.save_formatted_results(formatted_logs)
        return formatted_logs
