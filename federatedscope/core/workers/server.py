import logging
import copy
import os
import sys

import numpy as np
import pickle

from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.workers.base_server import BaseServer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Server(BaseServer):
    """
    The Server class, which describes the behaviors of server in an FL \
    course. The behaviors are described by the handled functions (named as \
    ``callback_funcs_for_xxx``).

    Arguments:
        ID: The unique ID of the server, which is set to 0 by default
        state: The training round
        config: the configuration
        data: The data owned by the server (for global evaluation)
        model: The model used for aggregation
        client_num: The (expected) client num to start the FL course
        total_round_num: The total number of the training round
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        aggregators: a protocol for aggregate all clients' model(s), see \
            ``federatedscope.core.aggregators``
        sample_client_num: number of client aggregated in each round
        msg_buffer: dict buffer for storing message
        staled_msg_buffer: list buffer for storing staled message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
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
                 unseen_clients_id=None,
                 **kwargs):
        super(Server, self).__init__(ID, state, config, model, strategy)
        # Register message handlers
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = dict()
        self.early_stopper = EarlyStopper(
            self._cfg.early_stop.patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)

        if self._cfg.federate.share_local_model \
                and not self._cfg.federate.process_num > 1:
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
            assert self.models is not None
            assert self.data is not None
            self.trainer = get_trainer(
                model=self.models[0],
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
        self.join_in_client_num = 0
        self.join_in_info = dict()
        # the unseen clients indicate the ones that do not contribute to FL
        # process by training on their local data and uploading their local
        # model update. The splitting is useful to check participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.unseen_clients_id = [] if unseen_clients_id is None \
            else unseen_clients_id

        # Server state
        self.is_finish = False

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

        # Initialize communication manager and message buffer
        self.msg_buffer = {'train': dict(), 'eval': dict()}
        self.staled_msg_buffer = list()
        if self.mode == 'standalone':
            comm_queue = kwargs.get('shared_comm_queue', None)
            if self._cfg.federate.process_num > 1:
                id2comm = kwargs.get('id2comm', None)
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue,
                    monitor=self._monitor,
                    id2comm=id2comm)
            else:
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            self.comm_manager = gRPCCommManager(host=host,
                                                port=port,
                                                client_num=client_num,
                                                cfg=self._cfg.distribute)
            logger.info('Server: Listen to {}:{}...'.format(host, port))

        # inject noise before broadcast
        self._noise_injector = None

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

    def run(self):
        """
        To start the FL course, listen and handle messages (for distributed \
        mode).
        """

        # Begin: Broadcast model parameters and start to FL train
        while self.join_in_client_num < self.client_num:
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg)

        # Running: listen for message (updates from clients),
        # aggregate and broadcast feedbacks (aggregated model parameters)
        min_received_num = self._cfg.asyn.min_received_num \
            if self._cfg.asyn.use else self._cfg.federate.sample_client_num
        num_failure = 0
        time_budget = self._cfg.asyn.time_budget if self._cfg.asyn.use else -1
        with Timeout(time_budget) as time_counter:
            while self.state <= self.total_round_num:
                try:
                    msg = self.comm_manager.receive()
                    move_on_flag = self.msg_handlers[msg.msg_type](msg)
                    if move_on_flag:
                        time_counter.reset()
                except TimeoutError:
                    logger.info('Time out at the training round #{}'.format(
                        self.state))
                    move_on_flag_eval = self.check_and_move_on(
                        min_received_num=min_received_num,
                        check_eval_result=True)
                    move_on_flag = self.check_and_move_on(
                        min_received_num=min_received_num)
                    if not move_on_flag and not move_on_flag_eval:
                        num_failure += 1
                        # Terminate the training if the number of failure
                        # exceeds the maximum number (default value: 10)
                        if time_counter.exceed_max_failure(num_failure):
                            logger.info(f'----------- Training fails at round '
                                        f'#{self.state}-------------')
                            break

                        # Time out, broadcast the model para and re-start
                        # the training round
                        logger.info(
                            f'----------- Re-starting the training round ('
                            f'Round #{self.state}) for {num_failure} time '
                            f'-------------')
                        # TODO: Clean the msg_buffer
                        if self.state in self.msg_buffer['train']:
                            self.msg_buffer['train'][self.state].clear()

                        self.broadcast_model_para(
                            msg_type='model_para',
                            sample_client_num=self.sample_client_num)
                    else:
                        num_failure = 0
                    time_counter.reset()

        self.terminate(msg_type='finish')

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """
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
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
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
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        return move_on_flag

    def check_and_save(self):
        """
        To save the results and save model after each evaluation, and check \
        whether to early stop.
        """

        # early stopping
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

        if should_stop:
            self._monitor.global_converged()
            self.comm_manager.send(
                Message(
                    msg_type="converged",
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    timestamp=self.cur_timestamp,
                    state=self.state,
                ))
            self.state = self.total_round_num + 1

        if should_stop or self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round or early stopped
            self.save_best_results()
            if not self._cfg.federate.make_global_eval:
                self.save_client_eval_results()
            self.terminate(msg_type='finish')

        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval:
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear()

        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1

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
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

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
            model.load_state_dict(merged_param, strict=False)

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
                    sample_client_num = aggregated_num + \
                                        self.dropout_num

                self.broadcast_model_para(msg_type='model_para',
                                          sample_client_num=sample_client_num)
                self.dropout_num = 0
        else:  # for synchronous training
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def _merge_and_format_eval_results(self):
        """
        The behaviors of server when receiving enough evaluating results
        """
        # Get all the message & aggregate
        formatted_eval_res = \
            self.merge_eval_results_from_all_clients()
        self.history_results = merge_dict_of_results(self.history_results,
                                                     formatted_eval_res)
        if self.mode == 'standalone' and \
                self._monitor.wandb_online_track and \
                self._monitor.use_wandb:
            self._monitor.merge_system_metrics_simulation_mode(
                file_io=False, from_global_monitors=True)
        self.check_and_save()

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
        save the evaluation results of each client when the fl course \
        early stopped or terminated
        """
        rnd = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][rnd]

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

    def merge_eval_results_from_all_clients(self):
        """
        Merge evaluation results from all clients, update best, \
        log the merged results and save them into eval_results.log

        Returns:
            the formatted merged results
        """
        round = max(self.msg_buffer['eval'].keys())
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

        formatted_logs_all_set = dict()
        for merge_type, eval_res_set in [("participated",
                                          eval_res_participated_clients),
                                         ("unseen", eval_res_unseen_clients)]:
            if eval_res_set != []:
                metrics_all_clients = dict()
                for client_eval_results in eval_res_set:
                    for key in client_eval_results.keys():
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(
                            float(client_eval_results[key]))
                formatted_logs = self._monitor.format_eval_res(
                    metrics_all_clients,
                    rnd=round,
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
                logger.info(formatted_logs)
                formatted_logs_all_set.update(formatted_logs)
                self._monitor.update_best_result(
                    self.best_results,
                    metrics_all_clients,
                    results_type="unseen_client_best_individual"
                    if merge_type == "unseen" else "client_best_individual")
                self._monitor.save_formatted_results(formatted_logs)
                for form in self._cfg.eval.report:
                    if form != "raw":
                        metric_name = form + "_unseen" if merge_type == \
                                                          "unseen" else form
                        self._monitor.update_best_result(
                            self.best_results,
                            formatted_logs[f"Results_{metric_name}"],
                            results_type=f"unseen_client_summarized_{form}"
                            if merge_type == "unseen" else
                            f"client_summarized_{form}")

        return formatted_logs_all_set

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast \
                behavior. And ``sample_client_num = -1`` denotes to \
                broadcast to all the clients.
            filter_unseen_clients: whether filter out the unseen clients that \
                do not contribute to FL process by training on their local \
                data and uploading their local model update. The splitting is \
                useful to check participation generalization gap in [ICLR'22, \
                What Do We Mean by Generalization in Federated Learning?] \
                You may want to set it to be False when in evaluation stage
        """
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
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        # quantization
        if msg_type == 'model_para' and not skip_broadcast and \
                self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_quantization
            nbits = self._cfg.quantization.nbits
            if self.model_num > 1:
                model_para = [
                    symmetric_uniform_quantization(x, nbits)
                    for x in model_para
                ]
            else:
                model_para = symmetric_uniform_quantization(model_para, nbits)

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def broadcast_client_address(self):
        """
        To broadcast the communication addresses of clients (used for \
        additive secret sharing)
        """

        self.comm_manager.send(
            Message(msg_type='address',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=self.comm_manager.get_neighbors()))

    def check_buffer(self,
                     cur_round,
                     min_received_num,
                     check_eval_result=False):
        """
        To check the message buffer

        Arguments:
            cur_round (int): The current round number
            min_received_num (int): The minimal number of the receiving \
                messages
            check_eval_result (bool): To check training results for \
                evaluation results

        Returns
            bool: Whether enough messages have been received or not
        """

        if check_eval_result:
            if 'eval' not in self.msg_buffer.keys() or len(
                    self.msg_buffer['eval'].keys()) == 0:
                return False

            buffer = self.msg_buffer['eval']
            cur_round = max(buffer.keys())
            cur_buffer = buffer[cur_round]
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
                return len(cur_buffer)+len(self.staled_msg_buffer) >= \
                       min_received_num

    def check_client_join_in(self):
        """
        To check whether all the clients have joined in the FL course.
        """

        if len(self._cfg.federate.join_in_info) != 0:
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """

        if self.check_client_join_in():
            if self._cfg.federate.use_ss or self._cfg.vertical.use:
                self.broadcast_client_address()

            # get sampler
            if 'client_resource' in self._cfg.federate.join_in_info:
                client_resource = [
                    self.join_in_info[client_index]['client_resource']
                    for client_index in np.arange(1, self.client_num + 1)
                ]
            else:
                if self._cfg.backend == 'torch':
                    try:
                        model_size = sys.getsizeof(pickle.dumps(
                            self.models[0])) / 1024.0 * 8.
                    except Exception as error:
                        model_size = 1.0
                        logger.warning(f'{error} in calculate model size.')
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

            # start feature engineering
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        """
        Interface for feature engineering, the default operation is none
        """
        trigger_train_func(**kwargs_for_trigger_train_func)

    def trigger_for_time_up(self, check_timestamp=None):
        """
        The handler for time up: modify the currency timestamp \
        and check the trigger condition
        """
        if self.is_finish:
            return False

        if check_timestamp is not None and \
                check_timestamp < self.deadline_for_cur_round:
            return False

        self.cur_timestamp = self.deadline_for_cur_round
        self.check_and_move_on()
        return True

    def terminate(self, msg_type='finish'):
        """
        To terminate the FL course
        """
        self.is_finish = True
        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.models[0].state_dict()

        self._monitor.finish_fl()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=model_para))

    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """

        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models;
            # for other cases such as ensemble, override the eval function
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
                    rnd=self.state,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_eval_res['Results_raw'],
                    results_type="server_global_eval")
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                self._monitor.save_formatted_results(formatted_eval_res)
                logger.info(formatted_eval_res)
            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate',
                                      filter_unseen_clients=False)

    def callback_funcs_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.

        Arguments:
            message: The received message.
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # dequantization
        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import \
                symmetric_uniform_dequantization
            if isinstance(content[1], list):  # multiple model
                sample_size = content[0]
                quant_model = [
                    symmetric_uniform_dequantization(x) for x in content[1]
                ]
            else:
                sample_size = content[0]
                quant_model = symmetric_uniform_dequantization(content[1])
            content = (sample_size, quant_model)

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

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
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)

        move_on_flag = self.check_and_move_on()
        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag

    def callback_funcs_for_join_in(self, message: Message):
        """
        The handling function for receiving the join in information. The \
        server might request for some information (such as \
        ``num_of_samples``) if necessary, assign IDs for the servers. \
        If all the clients have joined in, the training process will be \
        triggered.

        Arguments:
            message: The received message
        """

        if 'info' in message.msg_type:
            sender, info = message.sender, message.content
            for key in self._cfg.federate.join_in_info:
                assert key in info
            self.join_in_info[sender] = info
            logger.info('Server: Client #{:d} has joined in !'.format(sender))
        else:
            self.join_in_client_num += 1
            sender, address = message.sender, message.content
            if int(sender) == -1:  # assign number to client
                sender = self.join_in_client_num
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)
                self.comm_manager.send(
                    Message(msg_type='assign_client_id',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=str(sender)))
            else:
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)

            if len(self._cfg.federate.join_in_info) != 0:
                self.comm_manager.send(
                    Message(msg_type='ask_for_join_in_info',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=self._cfg.federate.join_in_info.copy()))

        self.trigger_for_start()

    def callback_funcs_for_metrics(self, message: Message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """

        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()

        self.msg_buffer['eval'][rnd][sender] = content

        return self.check_and_move_on(check_eval_result=True)

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
