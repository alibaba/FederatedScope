import logging
import copy
import os
import numpy as np
from collections import OrderedDict
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, gRPCCommManager
from federatedscope.core.monitors.monitor import update_best_result
from federatedscope.core.worker import Worker
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.worker.server import Server

logger = logging.getLogger(__name__)


class TextDTServer(Server):
    """
    The Server class, which describes the behaviors of server in an FL course.
    The attributes include:
        ID: The unique ID of the server, which is set to 0 by default
        state: The training round
        config: the configuration
        data: The data owned by the server (for global evaluation)
        model: The model used for aggregation
        client_num: The (expected) client num to start the FL course
        total_round_num: The total number of the training round
        device: The device to run local training and evaluation
        strategy: redundant attribute
    The behaviors are described by the handled functions (named as callback_funcs_for_xxx)
    """
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

    def check_and_move_on(self, check_eval_result=False):
        """
        To check the message_buffer, when enough messages are receiving, trigger some events (such as perform aggregation, evaluation, and move to the next training round)
        """

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
                            msg_list.append(train_msg_buffer[client_id])
                        else:
                            train_data_size, model_para_multiple = train_msg_buffer[
                                client_id]
                            msg_list.append((train_data_size,
                                             model_para_multiple[model_idx]))

                    # Trigger the monitor here (for training)
                    if 'dissim' in self._cfg.eval.monitoring:
                        B_val = self._monitor.calc_blocal_dissim(
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
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at round {:d}.'.
                        format(self.ID, self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round #{:d}/{:d}) -------------'
                        .format(self.state + 1, self._cfg.federate.total_round_num))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info(
                        'Server #{:d}: Training is finished! Starting evaluation.'
                        .format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()

    def merge_eval_results_from_all_clients(self, final_round=False):
        """
        Merge evaluation results from all clients,
        update best, log the merged results and save then into eval_results.log

        :param final_round:
        :return:
        """
        state = self.state if not final_round else self.state - 1
        eval_msg_buffer = self.msg_buffer['eval'][state]
        metrics_all_clients = dict()
        for each_client in eval_msg_buffer:
            client_eval_results = eval_msg_buffer[each_client]
            for key in client_eval_results.keys():
                res = client_eval_results[key]
                if isinstance(res, (dict, OrderedDict)):
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
            rnd=self.state,
            role='Server #',
            forms=self._cfg.eval.report)
        logger.info(formatted_logs)
        update_best_result(self.best_results,
                           metrics_all_clients,
                           results_type="client_individual",
                           round_wise_update_key=self._cfg.eval.
                           best_res_update_round_wise_key)
        self.save_formatted_results(formatted_logs)
        for form in self._cfg.eval.report:
            if form != "raw":
                update_best_result(self.best_results,
                                   formatted_logs[f"Results_{form}"],
                                   results_type=f"client_summarized_{form}",
                                   round_wise_update_key=self._cfg.eval.
                                   best_res_update_round_wise_key)
        return formatted_logs
