import os
import sys
import json
import logging
import copy
import pickle
import torch
import numpy as np
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.core.workers import Server
from federatedscope.nlp.trainer.utils import ContrastiveMonitor

logger = logging.getLogger(__name__)


class FedNLPServer(Server):
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

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         **kwargs)

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:  # in the training process
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                if aggregated_num == -1:
                    return move_on_flag

                self.state += 1
                self.aggregator.update_round(self.state)
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at the end of '
                        'round {:d}.'.format(self.ID, self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round '
                        '#{:d}/{:d}) -------------'.format(
                            self.state + 1,
                            self._cfg.federate.total_round_num))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()

        else:
            move_on_flag = False

        return move_on_flag

    def save_best_results(self):
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(
                os.path.join(self._cfg.federate.save_to, 'global_model.pt'),
                self.state)

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

    def trigger_for_start(self):
        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()

            # get sampler
            if 'client_resource' in self._cfg.federate.join_in_info:
                client_resource = [
                    self.join_in_info[client_index]['client_resource']
                    for client_index in np.arange(1, self.client_num + 1)
                ]
            else:
                if self._cfg.backend == 'torch':
                    model_size = sys.getsizeof(pickle.dumps(
                        self.model)) / 1024.0 * 8.
                else:
                    # TODO: calculate model size for TF Model
                    model_size = 1.0
                    logger.warning(f'The calculation of model size in backend:'
                                   f'{self._cfg.backend} is not provided.')

                client_resource = [
                    model_size / float(x['communication']) +
                    float(x['computation']) / 1000.
                    for x in self.client_resource_info
                ] if self.client_resource_info is not None else None

            if self.sampler is None:
                self.sampler = get_sampler(
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.client_num,
                    client_info=client_resource)

            # change the deadline if the asyn.aggregator is `time up`
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                self.deadline_for_cur_round = self.cur_timestamp + \
                                               self._cfg.asyn.time_budget

            logger.info('----------- Starting training (Round '
                        '#{:d}/{:d}) -------------'.format(
                            self.state + 1,
                            self._cfg.federate.total_round_num))
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def eval(self):
        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models; for other cases such as ensemble, override
            # the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation in server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)
                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state + 1,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                self._monitor.save_formatted_results(formatted_eval_res)
                logger.info(formatted_eval_res)
            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate')


class PFedNLPServer(FedNLPServer):
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

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         **kwargs)

        self.models = [
            copy.deepcopy(self.model) for _ in range(self.client_num)
        ]
        self.tasks = [
            config.model.pretrain_tasks[0]
            if config.model.pretrain_tasks else None
            for _ in range(self.client_num)
        ]
        self.aggregator.update_models(self.models)
        self.aggregator.update_neighbors(self.comm_manager.neighbors)
        if self._cfg.federate.restore_from != '':
            cur_round = self.aggregator.load_model(
                self._cfg.federate.restore_from)
            logger.info(
                "Restored the model from {}-th round's ckpt".format(cur_round))

    def _perform_federated_aggregation(self):
        train_msg_buffer = dict(
            sorted(self.msg_buffer['train'][self.state].items(),
                   key=lambda x: x[0]))
        msg_list = list()
        for client_id in train_msg_buffer:
            msg_list.append(train_msg_buffer[client_id])

        # Aggregate
        aggregated_num = len(msg_list)
        agg_info = {
            'client_feedback': msg_list,
            'recover_fun': self.recover_fun
        }
        avg_models, tasks = self.aggregator.aggregate(agg_info)
        self.tasks = tasks
        for i in range(self.client_num):
            self.models[i].load_state_dict(avg_models[i], strict=False)

        return aggregated_num

    def save_best_results(self):
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)

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
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receivers[i],
                        state=self.state,
                        content={
                            'model_para': model_para[i],
                            'task': self.tasks[i]
                        }))

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class PCFedNLPServer(FedNLPServer):
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

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         **kwargs)

        self.models = [
            copy.deepcopy(self.model) for _ in range(self.client_num)
        ]
        self.tasks = [
            config.model.pretrain_tasks[0]
            if config.model.pretrain_tasks else None
            for _ in range(self.client_num)
        ]
        self.aggregator.update_models(self.models)
        self.aggregator.update_neighbors(self.comm_manager.neighbors)

        self.contrast_monitor = ContrastiveMonitor()
        synth_feats, synth_toks = self._load_synth_data()
        self.contrast_monitor.update_enc_hidden(synth_feats)
        self.contrast_monitor.update_synth_tokens(synth_toks)
        self.aggregator.update_contrast_monitor(self.contrast_monitor)

        if self._cfg.federate.restore_from != '':
            cur_round = self.aggregator.load_model(
                self._cfg.federate.restore_from)
            logger.info(
                "Restored the model from {}-th round's ckpt".format(cur_round))

    def _load_synth_data(self):
        if self._cfg.data.debug:
            synth_dir = 'cache_debug/synthetic/'
        else:
            synth_dir = os.path.join(self._cfg.data.cache_dir, 'synthetic')
        synth_prim_weight = self._cfg.data.synth_prim_weight
        logger.info('Loading synthetic data from \'{}\''.format(synth_dir))
        with open(os.path.join(synth_dir, 'shapes.json')) as f:
            shapes = json.load(f)
        synth_feat_path = os.path.join(
            synth_dir, 'feature_{}.memmap'.format(synth_prim_weight))
        synth_tok_path = os.path.join(
            synth_dir, 'token_{}.memmap'.format(synth_prim_weight))
        synth_feats = np.memmap(filename=synth_feat_path,
                                shape=tuple(shapes['feature']),
                                mode='r',
                                dtype=np.float32)
        synth_toks = np.memmap(filename=synth_tok_path,
                               shape=tuple(shapes['token']),
                               mode='r',
                               dtype=np.int64)
        num_contrast = self._cfg.data.num_contrast
        synth_feats = {
            k: v
            for k, v in enumerate(
                torch.from_numpy(synth_feats)[:num_contrast])
        }
        synth_toks = {
            k: v
            for k, v in enumerate(torch.from_numpy(synth_toks)[:num_contrast])
        }
        return synth_feats, synth_toks

    def _perform_federated_aggregation(self):
        train_msg_buffer = dict(
            sorted(self.msg_buffer['train'][self.state].items(),
                   key=lambda x: x[0]))
        msg_list = list()
        for client_id in train_msg_buffer:
            msg_list.append(train_msg_buffer[client_id])

        # Aggregate
        aggregated_num = len(msg_list)
        agg_info = {
            'client_feedback': msg_list,
            'recover_fun': self.recover_fun
        }
        avg_models, tasks = self.aggregator.aggregate(agg_info)
        self.tasks = tasks
        if avg_models is not None and 'model_para' in avg_models:
            for i in range(self.client_num):
                self.models[i].load_state_dict(avg_models['model_para'][i],
                                               strict=False)

        if self.contrast_monitor.stat == 3:
            self.contrast_monitor.reset()
        if self._cfg.model.task != 'pretrain' and \
                self.contrast_monitor.stat == 2:
            self.msg_buffer['train'][self.state].clear()
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)
            return -1

        return aggregated_num

    def save_best_results(self):
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)

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
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receivers[i],
                        state=self.state,
                        content={
                            'model_para': model_para[i],
                            'task': self.tasks[i],
                            'contrast_monitor': self.contrast_monitor
                        }))

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')
