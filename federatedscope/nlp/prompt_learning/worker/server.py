import os
import sys
import logging
import pickle
import numpy as np
import torch
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.core.workers import Server
from federatedscope.nlp.prompt_learning.trainer.utils import merge_param_dict

logger = logging.getLogger(__name__)


class PLServer(Server):
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
            'client_feedback': [[x['sample_size'], x['model_para']]
                                for x in msg_list],
            'recover_fun': self.recover_fun,
        }
        avg_model = self.aggregator.aggregate(agg_info)
        merged_param = merge_param_dict(self.model.state_dict().copy(),
                                        avg_model)
        self.model.load_state_dict(merged_param, strict=False)

        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        skip_broadcast = self._cfg.federate.method in ['local', 'global']
        model_para = {} if skip_broadcast else self.model.state_dict()
        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=self.state,
                    content={'model_para': model_para}))

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')

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
                if self.state % self._cfg.eval.freq == 0 and \
                        self.state < self.total_round_num:
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
            # Perform training in server
            if self._cfg.federate.make_global_train:
                # model_para = self.model.state_dict()
                # self.trainer.update(model_para)
                sample_size, model_para, model_grads, train_metrics = \
                    self.trainer.train()
                # self.model.load_state_dict(model_para, strict=False)

                formatted_train_res = self._monitor.format_eval_res(
                    train_metrics,
                    rnd=self.state,
                    role='Server #',
                    return_raw=self._cfg.federate.make_global_eval)
                logger.info(formatted_train_res)

            # Evaluate on val dataset
            # model_para = self.model.state_dict()
            # self.trainer.update(model_para)
            val_metrics = self.trainer.evaluate(target_data_split_name='val')
            formatted_val_res = self._monitor.format_eval_res(
                val_metrics,
                rnd=self.state,
                role='Server #',
                return_raw=self._cfg.federate.make_global_eval)
            logger.info(formatted_val_res)

            comp_metric = f'val_{self._cfg.eval.metrics[0]}'
            max_score = -np.inf if self.state == 1 else \
                max(self.history_results['Results_raw'][comp_metric])
            cur_score = val_metrics[comp_metric]
            save_path = os.path.join(self._cfg.federate.pl_save_to,
                                     'best_model.pt')
            if cur_score > max_score:
                logger.info(f'Best score {cur_score} obtained. '
                            f'Model saved to {save_path}.')
                ckpt = {
                    'round': self.state,
                    'model': self.trainer.get_model_para(),
                    'val_score': cur_score
                }
                torch.save(ckpt, save_path)

            self.history_results = merge_dict_of_results(
                self.history_results, formatted_val_res)
            self._monitor.save_formatted_results(formatted_val_res)

            # Evaluate on test dataset
            if self.state == self.total_round_num:
                # last ckpt
                # model_para = self.model.state_dict()
                # self.trainer.update(model_para)
                test_metrics = self.trainer.evaluate(
                    target_data_split_name='test')
                formatted_test_res = self._monitor.format_eval_res(
                    test_metrics,
                    rnd=self.state,
                    role='Server # (Last)',
                    return_raw=self._cfg.federate.make_global_eval)
                self._monitor.save_formatted_results(formatted_test_res)
                logger.info(formatted_test_res)

                # best ckpt
                best_ckpt = torch.load(save_path, map_location='cpu')
                # model_para.update(best_ckpt['model'])
                self.trainer.update(best_ckpt['model'])
                logger.info(f"Loaded best model obtained in round "
                            f"{best_ckpt['round']} "
                            f"({best_ckpt['val_score']}).")

                test_metrics = self.trainer.evaluate(
                    target_data_split_name='test')
                formatted_test_res = self._monitor.format_eval_res(
                    test_metrics,
                    rnd=self.state,
                    role='Server # (Best)',
                    return_raw=self._cfg.federate.make_global_eval)
                self._monitor.save_formatted_results(formatted_test_res)
                logger.info(formatted_test_res)

            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate')
