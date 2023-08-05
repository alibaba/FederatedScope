import collections
import logging
import copy
import os
import subprocess
import time

import numpy as np
import torch
import wandb

from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.message import Message
from federatedscope.core.communication_standalone import StandaloneCommManager
from federatedscope.core.communication_grpc import gRPCCommManager
from federatedscope.core.workers import Worker
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict, Timeout, \
    merge_param_dict, get_time_stamp, get_time_interval, print_progress_bar
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from torch.onnx import TrainingMode
import matplotlib.pyplot as plt
import matplotlib

logger = logging.getLogger(__name__)

matplotlib.rcParams.update({
    'font.size': 25,
    'lines.linewidth': 3
})


class Server(Worker):
    """
    The Server class, which describes the behaviors of server in an FL course.
    The behaviors are described by the handled functions (named as
    callback_funcs_for_xxx).

    Arguments:
        ID: The unique ID of the server, which is set to 0 by default
        state: The training round
        config: the configuration
        data: The data owned by the server (for global evaluation)
        model: The model used for aggregation
        client_num: The (expected) client num to start the FL course
        total_round_num: The total number of the training round
        device: The device to run local training and evaluation
        strategy: redundant attribute
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
                 seen_clients_id=None,
                 unseen_clients_id=None,
                 **kwargs):

        super(Server, self).__init__(ID, state, config, model, strategy)

        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.early_stopper = EarlyStopper(
            self._cfg.early_stop.patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._cfg.early_stop.the_smaller_the_better)

        if self._cfg.federate.share_local_model:
            # put the model to the specified device
            model.to(device)
        # Build aggregator
        self.aggregator = get_aggregator(self._cfg.federate.method,
                                         model=model,
                                         device=device,
                                         online=self._cfg.federate.online_aggr,
                                         config=self._cfg)
        if self._cfg.federate.restore_from != '':
            if not os.path.exists(self._cfg.federate.restore_from):
                logger.warning(f'Invalid `restore_from`:'
                               f' {self._cfg.federate.restore_from}.')
            else:
                _ = self.aggregator.load_model(self._cfg.federate.restore_from)
                logger.info("Restored the model from {}-th round's ckpt")

        if int(config.model.model_num_per_trainer) != \
                config.model.model_num_per_trainer or \
                config.model.model_num_per_trainer < 1:
            raise ValueError(
                f"model_num_per_trainer should be integer and >= 1, "
                f"got {config.model.model_num_per_trainer}.")
        self.model_num = config.model.model_num_per_trainer
        self.models = [self.model]
        self.aggregators = [self.aggregator]
        if self.model_num > 1:
            self.models.extend(
                [copy.deepcopy(self.model) for _ in range(self.model_num - 1)])
            self.aggregators.extend([
                copy.deepcopy(self.aggregator)
                for _ in range(self.model_num - 1)
            ])

        # function for recovering shared secret
        self.recover_fun = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num)
        ).fixedpoint2float if self._cfg.federate.use_ss else None

        if self._cfg.federate.make_global_eval:
            # set up a trainer for conducting evaluation in server
            assert self.model is not None
            assert self.data is not None
            self.trainer = get_trainer(
                model=self.model,
                data=self.data,
                device=self.device,
                config=self._cfg,
                only_for_eval=True,
                monitor=self._monitor
            )  # the trainer is only used for global evaluation
            self.trainers = [self.trainer]
            if self.model_num > 1:
                # By default, the evaluation is conducted by calling
                # trainer[i].eval over all internal models
                self.trainers.extend([
                    copy.deepcopy(self.trainer)
                    for _ in range(self.model_num - 1)
                ])

        # Initialize the number of joined-in clients
        self._client_num = client_num
        self._total_round_num = total_round_num
        self.sample_client_num = int(self._cfg.federate.sample_client_num)
        # TODO: the number should be calculate in advance, PS: a test executor is required
        self.executor_num = self._cfg.federate.executor_num

        self.join_in_info = dict()
        # the unseen clients indicate the ones that do not contribute to FL
        # process by training on their local data and uploading their local
        # model update. The splitting is useful to check participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.seen_clients_id = seen_clients_id
        self.unseen_clients_id = unseen_clients_id
        assert len(set(self.seen_clients_id + self.unseen_clients_id)) == client_num

        # Server state
        self.is_finish = False
        # Early stopped
        self.is_stopped = False

        # Sampler
        if self._cfg.federate.sampler in ['uniform']:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=None)
        else:
            # Some type of sampler would be instantiated in trigger_for_start,
            # since they need more information
            self.sampler = None

        # Current Timestamp
        self.cur_timestamp = 0
        self.deadline_for_cur_round = 1

        # Staleness toleration
        self.staleness_toleration = self._cfg.asyn.staleness_toleration if \
            self._cfg.asyn.use else 0
        self.dropout_num = 0

        # Device information
        self.resource_info = kwargs['resource_info'] \
            if 'resource_info' in kwargs else None
        self.client_resource_info = kwargs['client_resource_info'] \
            if 'client_resource_info' in kwargs else None

        # Register message handlers
        self.msg_handlers = dict()
        self._register_default_handlers()

        # Initialize communication manager and message buffer
        self.msg_buffer = {'train': dict(), 'eval': dict()}
        self.staled_msg_buffer = list()
        if self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue']
            self.comm_manager = StandaloneCommManager(comm_queue=comm_queue,
                                                      monitor=self._monitor)
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            self.comm_manager = gRPCCommManager(host=host,
                                                port=port,
                                                client_num=client_num,
                                                save_path=self._cfg.outdir,
                                                compression=self._cfg.distribute.grpc_compress,
                                                model_type=self._cfg.model.type)
            logger.info('Server: Listen to {}:{}...'.format(host, port))

        # inject noise before broadcast
        self._noise_injector = None
        # TODO: actually it should be stored in executor_manager
        self.client_manager = {}

        # Maintain the best evaluation model here
        self.model_to_evaluate = dict()
        self.model_best_ever = None

        self.timestamp_start = None

        self.async_model_buffer = dict()

    @property
    def client_num(self):
        return self._client_num

    @client_num.setter
    def client_num(self, value):
        self._client_num = value

    @property
    def total_round_num(self):
        return self._total_round_num

    @total_round_num.setter
    def total_round_num(self, value):
        self._total_round_num = value

    def register_noise_injector(self, func):
        self._noise_injector = func

    def register_handlers(self, msg_type, callback_func):
        """
        To bind a message type with a handling function.

        Arguments:
            msg_type (str): The defined message type
            callback_func: The handling functions to handle the received
            message
        """
        self.msg_handlers[msg_type] = callback_func

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in)
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in)
        self.register_handlers('model_para', self.callback_funcs_model_para)
        self.register_handlers('metrics', self.callback_funcs_for_metrics)

    def run(self):
        """
        To start the FL course, listen and handle messages (for distributed
        mode).
        """

        start_rx_bytes = int(subprocess.getoutput("cat /sys/class/net/eth0/statistics/rx_bytes"))
        start_tx_bytes = int(subprocess.getoutput("cat /sys/class/net/eth0/statistics/tx_bytes"))
        start_time = time.time()

        logger.info(f"Begin with network traffic: RECEIVE: {start_rx_bytes} and SEND {start_tx_bytes}")

        # Begin: Broadcast model parameters and start to FL train
        while self.comm_manager.get_num_neighbors() < self.executor_num:
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg)

        # Running: listen for message (updates from clients),
        # aggregate and broadcast feedbacks (aggregated model parameters)
        num_failure = 0
        # time_budget = self._cfg.asyn.time_budget if self._cfg.asyn.use else -1
        # Notice: wait at most 15 minutes
        num_success = 0
        time_budget = self._cfg.federate.agg_time_budget
        if self._cfg.federate.agg_lower_bound == -1:
            agg_lower_bound = self._cfg.federate.min_received_num
        else:
            agg_lower_bound = self._cfg.federate.agg_lower_bound

        while self.state <= self.total_round_num:
            msg = self.comm_manager.receive()

            if msg.msg_type == "model_para":
                if self.state == self.total_round_num:
                    continue
                if self.is_stopped:
                    continue
                if time.time() - self.timestamp_start > time_budget:
                    # Timeout
                    logger.info(f"Time out ({time_budget}s) at the training round #{self.state}")
                    # Reset counter for success
                    num_success = 0
                    # Check if server receives enough training messages
                    move_on_flag = self.check_train_and_move_on(agg_lower_bound)
                    if not move_on_flag:
                        # not aggregation
                        if self._cfg.federate.upt_time_budget:
                            time_budget *= 2
                            logger.info(f"Time budget: [{time_budget / 2}, {time_budget}] at training round ${self.state}")
                        # Re-broadcast the model
                        num_failure += 1
                        logger.info(f"----------- Re-starting the training round "
                                    f"(Round #{self.state}) for {num_failure} time -------------")
                        self.broadcast_model_para(msg_type='model_para', sample_client_num=self.sample_client_num)
                else:
                    # Handle the message normally
                    move_on_flag = self.msg_handlers[msg.msg_type](msg)
                    if move_on_flag:
                        # After aggregation
                        num_success += 1
                        if self._cfg.federate.upt_time_budget and num_success >= 5:
                            num_success = 0
                            last_time_budget = time_budget
                            time_budget = max(5, time_budget - 5)
                            logger.info(f"Time budget: [{last_time_budget}, {time_budget}] at training round ${self.state}")
            else:
                # Evaluation messages
                self.msg_handlers[msg.msg_type](msg)

        end_time = time.time()
        end_rx_bytes = int(subprocess.getoutput("cat /sys/class/net/eth0/statistics/rx_bytes"))
        end_tx_bytes = int(subprocess.getoutput("cat /sys/class/net/eth0/statistics/tx_bytes"))

        rx_mb = (end_rx_bytes - start_rx_bytes) / 1024. / 1024.
        tx_mb = (end_tx_bytes - start_tx_bytes) / 1024. / 1024.
        train_time = end_time - start_time
        logger.info(f"Receive bytes: {rx_mb} MB | Send bytes: {tx_mb} MB | Time: {train_time} s")
        logger.info(f"Receive traffic: {rx_mb / train_time} MB/s | Send traffic: {tx_mb / train_time} MB/s")

        self.terminate(msg_type='finish')

    def check_train_and_move_on(self, min_received_num=None):
        # min_received_num must be decided by async or sync
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.min_received_num

        move_on_flag = True
        if self.check_buffer(self.state, min_received_num, False):
            # Receiving enough feedback in the training process
            aggregated_num = self._perform_federated_aggregation()

            self.state += 1

            # Delete the useless model in asyn mode
            if self._cfg.asyn.use:
                state_overtime = self.state - self._cfg.asyn.staleness_toleration - 1
                if state_overtime in self.async_model_buffer:
                    del self.async_model_buffer[state_overtime]

            # 此时self.state已经加1，得到的结果已经为下一轮的结果
            # 假设训练结束时，此数字为self.total_round_num-1，加1之后为==self.total_round_num
            if self.state % self._cfg.eval.freq == 0 and self.state < self.total_round_num:
                #  Evaluate during training
                logger.info(f'Server: Starting evaluation at the end '
                            f'of round {self.state - 1}.')
                self.eval()

            if self.state < self.total_round_num:
                # Move to next round of training
                logger.info(
                    f'----------- Starting a new training round (Round '
                    f'#{self.state}) -------------')
                # Clean the msg_buffer
                self.msg_buffer['train'][self.state - 1].clear()
                self.msg_buffer['train'][self.state] = dict()
                self.staled_msg_buffer.clear()
                # Start a new training round
                self._start_new_training_round(aggregated_num)
            else:
                # 正常训练结束
                self.eval()
        else:
            move_on_flag = False
        return move_on_flag

    def check_evaluate_and_move_on(self, round, min_received_num):
        """
        Min_received_num must be given as parameter
        """
        move_on_flag = True
        if self.check_buffer(round, min_received_num, True):
            self._merge_and_format_eval_results(round)
        else:
            move_on_flag = False
        return move_on_flag

    def check_and_save(self, cur_round):
        """
        To save the results and save model after each evaluation.
        """

        if self.is_stopped or cur_round == self.total_round_num:
            # 这里收到的就是最后一次所有client的evaluate的结果
            # 有两种情况，一种是early_stop之后，这里收到的最终一次的测试结果
            # 另一种是自然训练到结束，没有early_stop的情况
            self.save_client_eval_results()
            # 结束整个fl course
            self.state = self.total_round_num + 1
            logger.info(f"Current state is set to {self.state} in check_and_save function!")
        else:
            # 这里一定是训练过程中的测试结果，self.state<self.total_round_num
            # 判断是否要进行early stop
            if "Results_weighted_avg" in self.history_results and \
                    self._cfg.eval.best_res_update_round_wise_key in \
                    self.history_results['Results_weighted_avg']:
                should_stop = self.early_stopper.track_and_check(
                    self.history_results['Results_weighted_avg'][
                        self._cfg.eval.best_res_update_round_wise_key])
            elif "Results_avg" in self.history_results and \
                    self._cfg.eval.best_res_update_round_wise_key in \
                    self.history_results['Results_avg']:
                should_stop = self.early_stopper.track_and_check(
                    self.history_results['Results_avg'][
                        self._cfg.eval.best_res_update_round_wise_key])
            else:
                should_stop = False

            if should_stop and self.state != self.total_round_num:
                self._monitor.global_converged()
                # early stop
                # 将is_stopped设置成True，代表不再接收training的结果，因此不会进入callback_model_para，必须在这里启动最后一次的evaluation
                self.is_stopped = True
                self.state = self.total_round_num
                logger.info("".center(62, "="))
                logger.info("The training process is ended! Start global evaluation on all clients!")
                self.eval()

        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval:
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear()

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            staleness = list()

            for client_id in train_msg_buffer.keys():
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                staleness.append((client_id, self.state - state))

            # Trigger the monitor here (for training)
            if 'dissim' in self._cfg.eval.monitoring:
                # TODO: fix this
                B_val = self._monitor.calc_blocal_dissim(
                    model.load_state_dict(strict=False), msg_list)
                formatted_eval_res = self._monitor.format_eval_res(
                    B_val, rnd=self.state, role='Server #')
                logger.info(formatted_eval_res)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)

            model.load_state_dict(merged_param)

        return aggregated_num

    def _start_new_training_round(self, aggregated_num=0):
        """
        The behaviors for starting a new training round
        """
        if self._cfg.asyn.use:  # for asynchronous training
            if self._cfg.asyn.aggregator == "time_up":
                # Update the deadline according to the time budget
                self.deadline_for_cur_round = \
                    self.cur_timestamp + self._cfg.asyn.time_budget

            if self._cfg.asyn.broadcast_manner == \
                    'after_aggregating':
                if self._cfg.asyn.overselection:
                    sample_client_num = self.sample_client_num
                else:
                    sample_client_num = aggregated_num + self.dropout_num
                    if (self.state + 1) % 100 == 0:
                        sample_client_num += 1

                self.broadcast_model_para(msg_type='model_para',
                                          sample_client_num=sample_client_num)
                self.dropout_num = 0
        else:  # for synchronous training
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def _merge_and_format_eval_results(self, round):
        """
        做完一次evaluation之后在这里merge结果
        """
        # Get all the message & aggregate
        formatted_eval_res = \
            self.merge_eval_results_from_all_clients(round)
        self.history_results = merge_dict(self.history_results, formatted_eval_res)
        self.check_and_save(round)

    def save_best_results(self):
        """
        To Save the best evaluation results.
        """

        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)
        formatted_best_res = self._monitor.format_eval_res(
            results=self.best_results,
            rnd="Final",
            role='Server #',
            forms=["raw"],
            return_raw=True)
        logger.info(formatted_best_res)
        self._monitor.save_formatted_results(formatted_best_res)

    def save_client_eval_results(self):
        """
            save the evaluation results of each client when the fl course
            early stopped or terminated

        :return:
        """
        round = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][round]

        logger.info(f"Evaluation results on {len(eval_msg_buffer)} clients:")
        with open(os.path.join(self._cfg.outdir, "eval_results.log"),
                  "a") as outfile:
            for client_id, client_eval_results in eval_msg_buffer.items():
                formatted_res = self._monitor.format_eval_res(
                    client_eval_results,
                    rnd=self.state,
                    role='Client #{}'.format(client_id),
                    return_raw=True)
                logger.info(formatted_res)
                outfile.write(str(formatted_res) + "\n")

    def merge_eval_results_from_all_clients(self, round):
        """
            Merge evaluation results from all clients, update best,
            log the merged results and save them into eval_results.log

        :returns: the formatted merged results
        """

        eval_msg_buffer = self.msg_buffer['eval'][round]
        eval_res_participated_clients = []
        eval_res_unseen_clients = []
        for client_id in eval_msg_buffer:
            if eval_msg_buffer[client_id] is None:
                continue
            if client_id in self.unseen_clients_id:
                eval_res_unseen_clients.append(eval_msg_buffer[client_id])
            else:
                eval_res_participated_clients.append(
                    eval_msg_buffer[client_id])

        if len(eval_res_unseen_clients) == 0:
            metrics = [("participated", eval_res_participated_clients)]
        else:
            # Evaluation at the end
            metrics = [
                ("participated", eval_res_participated_clients),
                ("unseen", eval_res_unseen_clients),
                ("all", list(eval_msg_buffer.values()))
            ]

        formatted_logs_all_set = dict()
        for merge_type, eval_res_set in metrics:
            if eval_res_set != []:
                metrics_all_clients = dict()
                for client_eval_results in eval_res_set:
                    for key in client_eval_results.keys():
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(
                            float(client_eval_results[key]))
                if self.state == self.total_round_num:
                    round_log = f"Final_{merge_type}"
                else:
                    round_log = round
                formatted_logs = self._monitor.format_eval_res(
                    metrics_all_clients,
                    rnd=round_log,
                    role='Server #',
                    forms=self._cfg.eval.report)
                if merge_type == "unseen":
                    for key, val in copy.deepcopy(formatted_logs).items():
                        if isinstance(val, dict):
                            # to avoid the overrides of results using the
                            # same name, we use new keys with postfix `unseen`:
                            # 'Results_weighted_avg' ->
                            # 'Results_weighted_avg_unseen'
                            formatted_logs[key + "_unseen"] = val
                            del formatted_logs[key]
                logger.info(f"{merge_type}: {formatted_logs}")
                formatted_logs_all_set.update(formatted_logs)

                self._monitor.update_best_result(
                    self.best_results,
                    metrics_all_clients,
                    results_type="unseen_client_best_individual"
                    if merge_type == "unseen" else "client_best_individual",
                    round_wise_update_key=self._cfg.eval.
                    best_res_update_round_wise_key)

                self._monitor.save_formatted_results(formatted_logs)
                for form in self._cfg.eval.report:
                    if form != "raw":
                        metric_name = form + "_unseen" if merge_type == "unseen" else form
                        update_best_result = self._monitor.update_best_result(
                            self.best_results,
                            formatted_logs[f"Results_{metric_name}"],
                            results_type=f"unseen_client_summarized_{form}"
                            if merge_type == "unseen" else
                            f"client_summarized_{form}",
                            round_wise_update_key=self._cfg.eval.
                            best_res_update_round_wise_key)
                        if form != "weighted_avg":
                            update_best_result = False

        if update_best_result:
            # Update the best model
            self.model_best_ever = self.model_to_evaluate[round]
            # Remove the evaluation model
            del self.model_to_evaluate[round]

        return formatted_logs_all_set

    def broadcast_model_para(self,
                             sample_client_num,
                             msg_type='model_para',
                             filter_unseen_clients=True,
                             model_dict=None):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num:
            filter_unseen_clients: whether filter out the unseen clients that
                do not contribute to FL process by training on their local
                data and uploading their local model update. The splitting is
                useful to check participation generalization gap in [ICLR'22,
                What Do We Mean by Generalization in Federated Learning?]
                You may want to set it to be False when in evaluation stage
            model_dict:
        """
        assert msg_type in ["model_para", "evaluate"], f"msg_type cannot be {msg_type}"

        # Load the best model to evaluate
        if model_dict is not None:
            self.model.load_state_dict(model_dict)

        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        # Sample clients
        if msg_type == "model_para":
            # client_id_choice = self.sampler.sample(size=sample_client_num, state="working")
            client_id_choice = self.sampler.sample(size=sample_client_num, state=None)
        else:
            if sample_client_num == self.client_num:
                client_id_choice = [_ for _ in range(1, 1 + self.client_num)]
            else:
                client_id_choice = np.random.choice(self.seen_clients_id, sample_client_num, replace=False).tolist()

        # Sample executors
        sample_executor_num = min(len(client_id_choice), self.comm_manager.get_num_neighbors())
        executor_choice = self.comm_manager.sample_neighbor(sample_executor_num)
        # Dispatch clients into different executors
        assert len(client_id_choice) >= len(executor_choice)
        group_client = [_.tolist() for _ in np.array_split(client_id_choice, sample_executor_num)]

        # Prepare MNN model
        timestamp = get_time_stamp("%Y-%m-%d_%H-%M-%S")
        path_onnx = os.path.join(self._cfg.outdir, f"{timestamp}_{self.model.__class__.__name__}_round-{self.state}.onnx")
        path_mnn = os.path.join(self._cfg.outdir, f"{timestamp}_{self.model.__class__.__name__}_round-{self.state}.mnn")

        # TODO: image size and channel
        input_shape = self._cfg.model.input_shape

        # TODO: 将模型load进来一部分
        # 如果是第0轮，全部传过去；如果是第一轮之后，则用ConvNet2Body加载模型，然后传到模型中去

        if self._cfg.federate.method.lower() == "fedbabu":
            if self.state > 0:
                if self._cfg.model.type.lower() == "convnet2":
                    from federatedscope.cv.model.cnn import ConvNet2Body
                    if self._cfg.data.type.lower() == "femnist":
                        class_num = 62
                    else:
                        raise NotImplementedError(f"Unknown num_class for {self._cfg.data.type.lower()}")
                    model_body = ConvNet2Body(input_shape[0], input_shape[1], input_shape[2], 1024, class_num=class_num)
                    state_dict = self.model.state_dict()
                    model_body.load_state_dict({_: state_dict[_] for _ in model_body.state_dict().keys()})
                    pass_model = model_body
                elif self._cfg.model.type.lower() == "convnet5":
                    from federatedscope.cv.model.cnn import ConvNet5Body
                    model_body = ConvNet5Body(input_shape[0], input_shape[1], input_shape[2], 512, class_num=self._cfg.model.out_channels)
                    state_dict = self.model.state_dict()
                    model_body.load_state_dict({_: state_dict[_] for _ in model_body.state_dict().keys()})
                    pass_model = model_body
                else:
                    raise NotImplementedError(f"Not support {self._cfg.model.type}")
            else:
                pass_model = self.model
        else:
            pass_model = self.model

        dummy_input = torch.randn(size=[1] + list(input_shape), device=torch.device("cpu"))
        input_names = ["input"]
        output_names = ["output"]
        # logger.info(f"Creat ONNX model in {path_onnx}")
        torch.onnx.export(pass_model, dummy_input, path_onnx, verbose=False, training=TrainingMode.TRAINING, input_names=input_names,
                          do_constant_folding=False, opset_version=12,
                          output_names=output_names)

        # Check if the onnx file exists
        if not os.path.exists(path_onnx):
            raise FileNotFoundError(f"Cannot found ONNX file in {path_onnx}.")

        # logger.info(f"Convert ONNX model into MNN model in {path_mnn}")

        if self._cfg.distribute.mnn_compress == "fp16":
            logger.info("Enable fp16 quantization in mnn convert")
            cmd_convert = f"{self._cfg.mnn.cmd_convert} -f ONNX --modelFile {path_onnx} --MNNModel {path_mnn} --bizCode biz --forTraining true --fp16 > /dev/null"
        elif self._cfg.distribute.mnn_compress == "int8":
            logger.info("Enable int8 quantization in mnn convert")
            cmd_convert = f"{self._cfg.mnn.cmd_convert} -f ONNX --modelFile {path_onnx} --MNNModel {path_mnn} --bizCode biz --forTraining true --weightQuantBits 8 --weightQuantAsymmetric > /dev/null"
        else:
            cmd_convert = f"{self._cfg.mnn.cmd_convert} -f ONNX --modelFile {path_onnx} --MNNModel {path_mnn} --bizCode biz --forTraining true > /dev/null"
        os.system(cmd_convert)
        # logger.info("Succeeded!")

        if not os.path.exists(path_mnn):
            raise FileNotFoundError(f"Cannot found MNN file in {path_mnn}.")

        # send file
        logger.info(f"Round: {self.state}, Event: {msg_type}, #{len(client_id_choice)} Participated  Clients: {client_id_choice}")
        self.comm_manager.send_file(msg_type, self.state, path_mnn, executor_choice, group_client)

        # delete onnx and mnn
        # logger.info(f"Delete ONNX model in {path_onnx}")
        os.remove(path_onnx)
        # logger.info(f"Delete MNN model in {path_mnn}")
        os.remove(path_mnn)

        # Store model in async mode
        if self._cfg.asyn.use:
            self.async_model_buffer[self.state] = copy.deepcopy(self.model.state_dict())

        if msg_type == "model_para":
            # reset timestamp for training event
            self.timestamp_start = time.time()

        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def broadcast_client_address(self):
        """
        To broadcast the communication addresses of clients (used for
        additive secret sharing)
        """

        self.comm_manager.send(
            Message(msg_type='address',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=get_time_stamp(),
                    content=self.comm_manager.get_neighbors()))

    def check_buffer(self,
                     cur_round,
                     min_received_num,
                     check_eval_result=False):
        """
        To check the message buffer

        Arguments:
        cur_round (int): The current round number
        min_received_num (int): The minimal number of the receiving messages
        check_eval_result (bool): To check training results for evaluation
        results
        :returns: Whether enough messages have been received or not
        :rtype: bool
        """

        if check_eval_result:
            if 'eval' not in self.msg_buffer.keys() or len(
                    self.msg_buffer['eval'].keys()) == 0:
                return False

            buffer = self.msg_buffer['eval']
            cur_buffer = buffer[cur_round]
            logger.info(f"Received {len(cur_buffer)}/{min_received_num} eval msgs in Round #{cur_round}! ({len(cur_buffer) >= min_received_num})")
            return len(cur_buffer) >= min_received_num
        else:
            if cur_round not in self.msg_buffer['train']:
                cur_buffer = dict()
            else:
                cur_buffer = self.msg_buffer['train'][cur_round]
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                if self.cur_timestamp >= self.deadline_for_cur_round and len(
                        cur_buffer) + len(self.staled_msg_buffer) == 0:
                    # When the time budget is run out but the server has not
                    # received any feedback
                    logger.warning(
                        f'The server has not received any feedback when the '
                        f'time budget has run out, therefore the server would '
                        f'wait for more {self._cfg.asyn.time_budget} seconds. '
                        f'Maybe you should carefully reset '
                        f'`cfg.asyn.time_budget` to a reasonable value.')
                    self.deadline_for_cur_round += self._cfg.asyn.time_budget
                    if self._cfg.asyn.broadcast_manner == \
                            'after_aggregating' and self.dropout_num != 0:
                        self.broadcast_model_para(
                            msg_type='model_para',
                            sample_client_num=self.dropout_num)
                        self.dropout_num = 0
                return self.cur_timestamp >= self.deadline_for_cur_round
            else:
                return len(cur_buffer) + len(self.staled_msg_buffer) >= \
                       min_received_num

    def check_client_join_in(self):
        """
        To check whether all the clients have joined in the FL course.
        """

        return self.comm_manager.get_num_neighbors() == self.executor_num

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """

        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()

            assert self.sampler is not None, f"Sampler cannot not None"

            # change the deadline if the asyn.aggregator is `time up`
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                self.deadline_for_cur_round = self.cur_timestamp + \
                                              self._cfg.asyn.time_budget

            # Print the distribution of the clients
            device_info = dict()
            device_list = dict()
            for client_id, info in self.client_manager.items():
                for key, value in info.items():
                    if key not in device_info:
                        device_info[key] = collections.defaultdict(int)
                    device_info[key][value] += 1
                    if key not in device_list:
                        device_list[key] = list()
                    device_list[key].append(value)

            wandb_dict = {}
            for key, info in device_list.items():
                plt.figure(figsize=(8, 6))
                if key.lower() == "latency":
                    plt.xlabel("Latency (ms)")
                elif key.lower() == "network":
                    plt.xlabel("Network type")
                elif key.lower() == "num_cpu_cores":
                    plt.xlabel("Number of cpu cores")
                    plt.xticks([1, 2, 3, 4], [1, 2, 3, 4])
                else:
                    plt.xlabel(f"{key}")

                plt.ylabel("Number of clients")
                plt.hist(info)
                plt.title(f"Distribution of {key}")
                plt.grid(linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.savefig(f"{self._cfg.outdir}/distribution-{key}.png")
                if key.lower() in ["num_cpu_cores", "latency", "network"]:
                    wandb_dict[key] = wandb.Image(f"{self._cfg.outdir}/distribution-{key}.png")

            if self._cfg.wandb.use:
                wandb.log(wandb_dict)

            # Print table
            len_name, len_number = 59, 8
            for key, info in device_info.items():
                if key in ["CPU_ABI2", "DISPLAY"]:
                    continue

                logger.info("+" + "=" * len_name + "+" + "=" * len_number + "+")
                logger.info("|\033[1;37m" + key.center(len_name, " ") + "\033[0m| \033[1;37mNumber\033[0m |")
                logger.info("+" + "=" * len_name + "+" + "=" * len_number + "+")
                for name, number in info.items():
                    logger.info("|" + str(name).center(len_name, " ") + "|" + str(number).center(8, " ") + "|")
                    logger.info("+" + "-" * len_name + "+" + "-" * len_number + "+")

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))

            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def trigger_for_time_up(self, check_timestamp=None):
        """
        The handler for time up: modify the currency timestamp
        and check the trigger condition
        """
        if self.is_finish:
            return False

        if check_timestamp is not None and \
                check_timestamp < self.deadline_for_cur_round:
            return False

        self.cur_timestamp = self.deadline_for_cur_round
        self.check_train_and_move_on()
        return True

    def terminate(self, msg_type='finish'):
        """
        To terminate the FL course
        """
        self.is_finish = True

        self._monitor.finish_fl()

        self.comm_manager.close()

    def eval(self):
        """
        To conduct evaluation. When cfg.federate.make_global_eval=True,
        a global evaluation is conducted by the server.
        """
        if self.state >= self.total_round_num or self.is_finish:
            # Evaluation at the end of training
            sample_client_num = self.client_num
            # Evaluate the best model here on all the clients
            self.broadcast_model_para(msg_type='evaluate',
                                      sample_client_num=sample_client_num,
                                      filter_unseen_clients=False,
                                      model_dict=self.model_best_ever)
        else:
            # Record the current model for evaluate
            self.model_to_evaluate[self.state] = copy.deepcopy(self.model.state_dict())
            # Evaluation during training
            sample_client_num = self.sample_client_num
            self.broadcast_model_para(msg_type='evaluate',
                                      sample_client_num=sample_client_num,
                                      filter_unseen_clients=True)

    def callback_funcs_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers
            check_and_move_on (perform aggregation when enough feedback has
            been received).
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message, which includes sender, receiver,
                state, and content. More detail can be found in
                federatedscope.core.message
        """
        if self.is_finish:
            return 'finish'
        if self.is_stopped:
            return 'stopped'

        round = message.state
        sender = message.sender
        content = message.content

        self.sampler.change_state(sender, 'idle')
        if round == self.state or round >= self.state - self.staleness_toleration:

            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.min_received_num
            # compute the arrived time
            print_progress_bar(len(self.msg_buffer['train'].get(round, list())) + 1, min_received_num)

            # Compute delta in async mode
            if self._cfg.asyn.use:
                n_samples, params_new = content
                params_old = self.async_model_buffer[round]
                for key, param_new in params_new.items():
                    params_new[key] = param_new - params_old[key]
                content = (n_samples, params_new)

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content))
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from {sender} in round #{round}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        move_on_flag = self.check_train_and_move_on()
        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag

    def callback_funcs_for_join_in(self, message: Message):
        """
        The handling function for receiving the join in information. The
        server might request for some information (such as num_of_samples)
        if necessary, assign IDs for the servers.
        If all the clients have joined in, the training process will be
        triggered.

        Arguments:
            message: The received message
        """

        sender, content = message.sender, message.content
        address = {"host": content["host"], "port": content["port"]}
        # Drop this message if the client already join in fl course
        if f"{content['host']}:{content['port']}" in self.comm_manager.neighbors.values() or self.comm_manager.get_num_neighbors() + 1 > self.executor_num:
            return
        else:
            logger.info(f"{self.comm_manager.get_num_neighbors() + 1}/{self.executor_num} executors have joined in from {content['host']}:{content['port']}")
            if int(sender) == -1:
                # Index the executors from 1
                sender = self.comm_manager.get_new_id()
                self.comm_manager.add_neighbors(neighbor_id=sender, address=address)
                # Assign id
                self.comm_manager.send(
                    Message(msg_type='assign_executor_id',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=get_time_stamp(),
                            content=str(sender)))
            else:
                raise RuntimeError(f"Sender cannot be {sender}")

            self.client_manager[sender] = {key: value for key, value in content.items() if
                                           key not in ["host", "port"]}

            self.trigger_for_start()

    def callback_funcs_for_metrics(self, message: Message):
        """
        The handling function for receiving the evaluation results,
        which triggers check_and_move_on
            (perform aggregation when enough feedback has been received).

        Arguments:
            message: The received message
        """

        round = message.state
        sender = message.sender
        content = message.content

        if round not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][round] = dict()

        self.msg_buffer['eval'][round][sender] = content

        # Decide the number of receive metrics
        if round >= self.total_round_num:
            min_received_num = self.client_num - 100
        else:
            min_received_num = self.sample_client_num
        return self.check_evaluate_and_move_on(round, min_received_num=min_received_num)
