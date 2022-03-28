import logging
import copy

import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, gRPCCommManager
from federatedscope.core.worker import Worker
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.utils import formatted_logging
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.monitor import calc_blocal_dissim
from federatedscope.core.secret_sharing import AdditiveSecretSharing


class Server(Worker):
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

        super(Server, self).__init__(ID, state, config, model, strategy)

        self.data = data
        self.device = device
        self.best_results = dict()
        self.history_results = []  # Record the results

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
            cur_round = self.aggregator.load_model(
                self._cfg.federate.restore_from)
            logging.info("Restored the model from {}-th round's ckpt")

        if int(config.model.model_num_per_trainer) != config.model.model_num_per_trainer or \
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
                only_for_eval=True
            )  # the trainer is only used for global evaluation
            self.trainers = [self.trainer]
            if self.model_num > 1:
                # By default, the evaluation is conducted by calling trainer[i].eval over all internal models
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

        # Register message handlers
        self.msg_handlers = dict()
        self._register_default_handlers()

        # Initialize communication manager and message buffer
        self.msg_buffer = {'train': dict(), 'eval': dict()}
        if self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue']
            self.comm_manager = StandaloneCommManager(comm_queue=comm_queue)
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            self.comm_manager = gRPCCommManager(host=host,
                                                port=port,
                                                client_num=client_num)
            logging.info('Server #{:d}: Listen to {}:{}...'.format(
                self.ID, host, port))

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

    def register_handlers(self, msg_type, callback_func):
        self.msg_handlers[msg_type] = callback_func

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in)
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in)
        self.register_handlers('model_para', self.callback_funcs_model_para)
        self.register_handlers('metrics', self.callback_funcs_for_metrics)

    def run(self):
        # Begin: Broadcast model parameters and start to FL train
        while self.join_in_client_num < self.client_num:
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg)

        # Running: listen for message (updates from clients), aggregate and broadcast feedbacks (aggregated model parameters)
        while self.state <= self.total_round_num:
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg)

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.model.state_dict()

        self.comm_manager.send(
            Message(msg_type='finish',
                    sender=self.ID,
                    receiver=None,
                    state=self.state,
                    content=model_para))

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
                            msg_list.append(train_msg_buffer[client_id])
                        else:
                            train_data_size, model_para_multiple = train_msg_buffer[
                                client_id]
                            msg_list.append((train_data_size,
                                             model_para_multiple[model_idx]))


                    # Trigger the monitor here (for training)
                    if 'dissim' in self._cfg.monitoring:
                        B_val = calc_blocal_dissim(
                            model.load_state_dict(strict=False), msg_list)
                        formatted_logs = formatted_logging(B_val,
                                                           rnd=self.state,
                                                           role='Server #')
                        logging.info(formatted_logs)

                    # Aggregate
                    agg_info = {
                        'client_feedback': msg_list,
                        'recover_fun': self.recover_fun
                    }
                    result = aggregator.aggregate(agg_info)
                    model.load_state_dict(result, strict=False)
                    #aggregator.update(result)

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    #  Evaluate
                    logging.info(
                        'Server #{:d}: Starting evaluation at round {:d}.'.
                        format(self.ID, self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logging.info(
                        '----------- Starting a new training round (Round #{:d}) -------------'
                        .format(self.state))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logging.info(
                        'Server #{:d}: Training is finished! Starting evaluation.'
                        .format(self.ID))
                    self.eval()
                    if self._cfg.federate.save_to != '':
                        self.aggregator.save_model(self._cfg.federate.save_to,
                                                   self.state)
                    formatted_best_res = formatted_logging(self.best_results,
                                                           rnd="Final",
                                                           role='Server #',
                                                           forms=["raw"])
                    logging.info(formatted_best_res)

            else:  # in the evaluation process
                # Get all the message & aggregate
                eval_msg_buffer = self.msg_buffer['eval'][self.state]
                metrics_all_clients = dict()
                for each_client in eval_msg_buffer:
                    client_eval_results = eval_msg_buffer[each_client]
                    for key in client_eval_results.keys():
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(
                            float(client_eval_results[key]))

                formatted_logs = formatted_logging(metrics_all_clients,
                                                   rnd=self.state,
                                                   role='Server #',
                                                   forms=self._cfg.eval.report)
                self.history_results.append(formatted_logs)
                logging.info(formatted_logs)
                self.update_best_result(metrics_all_clients,
                                        results_type="client_individual")
                for form in self._cfg.eval.report:
                    if form != "raw":
                        self.update_best_result(
                            formatted_logs[f"Results_{form}"],
                            results_type=f"client_summarized_{form}")

                if self.state == self.total_round_num:
                    #break out the loop for distributed mode
                    self.state += 1

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1):
        if sample_client_num > 0:
            receiver = np.random.choice(np.arange(1, self.client_num + 1),
                                        size=sample_client_num,
                                        replace=False).tolist()
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.model.state_dict()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=self.state,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

    def broadcast_client_address(self):
        self.comm_manager.send(
            Message(msg_type='address',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    content=self.comm_manager.get_neighbors()))

    def check_buffer(self, cur_round, minimal_number, check_eval_result=False):
        if check_eval_result:
            if 'eval' not in self.msg_buffer.keys():
                return False
            buffer = self.msg_buffer['eval']
        else:
            buffer = self.msg_buffer['train']

        if cur_round not in buffer or len(buffer[cur_round]) < minimal_number:
            return False
        else:
            return True

    def check_client_join_in(self):
        if len(self._cfg.federate.join_in_info) != 0:
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num

    def trigger_for_start(self):
        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def update_best_result(self, results, results_type):
        if isinstance(results, dict):
            if results_type not in self.best_results:
                self.best_results[results_type] = dict()
            best_result = self.best_results[results_type]
            for key in results:
                cur_result = results[key]
                if 'loss' in key or 'std' in key:  # the smaller, the better
                    if results_type == "client_individual":
                        cur_result = min(cur_result)
                    if key not in best_result or cur_result < best_result[key]:
                        best_result[key] = cur_result
                        logging.info(
                            f"Find new best result for {results_type}.{key} with value {cur_result}"
                        )
                elif 'acc' in key:  # the larger, the better
                    if results_type == "client_individual":
                        cur_result = max(cur_result)
                    if key not in best_result or cur_result > best_result[key]:
                        best_result[key] = cur_result
                        logging.info(
                            f"Find new best result for {results_type}.{key} with value {cur_result}"
                        )
                else:
                    # unconcerned metric
                    pass
        else:
            raise ValueError

    def eval(self):
        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation in server
                metrics = trainer.evaluate()
                formatted_logs = formatted_logging(metrics,
                                                   rnd=self.state,
                                                   role='Global-Eval-Server #',
                                                   forms=self._cfg.eval.report)
                self.update_best_result(formatted_logs['Results_raw'],
                                        results_type="server_global_eval")
                self.history_results.append(formatted_logs)
                logging.info(formatted_logs)
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate')

    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = content

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content)
        self.check_and_move_on()

    def callback_funcs_for_join_in(self, message: Message):
        if 'info' in message.msg_type:
            sender, info = message.sender, message.content
            for key in self._cfg.federate.join_in_info:
                assert key in info
            self.join_in_info[sender] = info
            logging.info('Server #{:d}: Client #{:d} has joined in !'.format(
                self.ID, sender))
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
                            content=self._cfg.federate.join_in_info.copy()))

        self.trigger_for_start()

    def callback_funcs_for_metrics(self, message: Message):
        round, sender, content = message.state, message.sender, message.content

        if round not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][round] = dict()

        self.msg_buffer['eval'][round][sender] = content

        self.check_and_move_on(check_eval_result=True)
