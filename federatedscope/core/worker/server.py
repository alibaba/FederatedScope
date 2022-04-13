import logging
import copy
import os

import numpy as np

from federatedscope.core.early_stopper import EarlyStopper
from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, gRPCCommManager
from federatedscope.core.worker import Worker
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.utils import formatted_logging, merge_dict
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.monitor import calc_blocal_dissim
from federatedscope.core.secret_sharing import AdditiveSecretSharing


class Server(Worker):
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
        """
        To bind a message type with a handled function
        """
        self.msg_handlers[msg_type] = callback_func

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in)
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in)
        self.register_handlers('model_para', self.callback_funcs_model_para)
        self.register_handlers('metrics', self.callback_funcs_for_metrics)

    def run(self):
        """
        To start the FL course, listen and handle messages (for distributed mode)
        """

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
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    content=model_para))

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
                        B_val = calc_blocal_dissim(
                            model.load_state_dict(strict=False), msg_list)
                        formatted_eval_res = formatted_logging(B_val,
                                                               rnd=self.state,
                                                               role='Server #')
                        logging.info(formatted_eval_res)

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

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()

    def check_and_save(self):
        """
        To save the results and save model after each evaluation
        """
        # early stopping
        should_stop = False

        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in self.history_results['Results_weighted_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        elif "Results_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in self.history_results['Results_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        else:
            should_stop = False

        if should_stop:
            self.state = self.total_round_num + 1

        if should_stop or self.state == self.total_round_num:
            logging.info(
                'Server #{:d}: Final evaluation is finished! Starting merging results.'
                .format(self.ID))
            # last round
            # TODO: call save with a specified frequency
            if self._cfg.federate.save_to != '':
                self.aggregator.save_model(self._cfg.federate.save_to,
                                           self.state)
            formatted_best_res = formatted_logging(self.best_results,
                                                   rnd="Final",
                                                   role='Server #',
                                                   forms=["raw"])
            with open(os.path.join(self._cfg.outdir, "eval_results.log"),
                      "a") as outfile:
                outfile.write(str(formatted_best_res) + "\n")
            logging.info(formatted_best_res)

            if self.model_num > 1:
                model_para = [model.state_dict() for model in self.models]
            else:
                model_para = self.model.state_dict()
            self.comm_manager.send(
                Message(msg_type='finish',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        content=model_para))

        if self.state == self.total_round_num:
            #break out the loop for distributed mode
            self.state += 1

    def merge_eval_results_from_all_clients(self, final_round=False):
        state = self.state if not final_round else self.state - 1
        eval_msg_buffer = self.msg_buffer['eval'][state]
        metrics_all_clients = dict()
        for each_client in eval_msg_buffer:
            client_eval_results = eval_msg_buffer[each_client]
            for key in client_eval_results.keys():
                if key not in metrics_all_clients:
                    metrics_all_clients[key] = list()
                metrics_all_clients[key].append(float(
                    client_eval_results[key]))
        formatted_logs = formatted_logging(metrics_all_clients,
                                           rnd=self.state,
                                           role='Server #',
                                           forms=self._cfg.eval.report)
        logging.info(formatted_logs)
        self.update_best_result(metrics_all_clients,
                                results_type="client_individual",
                                round_wise_update_key=self._cfg.eval.
                                best_res_update_round_wise_key)
        with open(os.path.join(self._cfg.outdir, "eval_results.log"),
                  "a") as outfile:
            outfile.write(str(formatted_logs) + "\n")
        for form in self._cfg.eval.report:
            if form != "raw":
                self.update_best_result(
                    formatted_logs[f"Results_{form}"],
                    results_type=f"client_summarized_{form}",
                    round_wise_update_key=self._cfg.eval.
                    best_res_update_round_wise_key)
        return formatted_logs

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1):
        """
        To broadcast the message to all clients or sampled clients
        """
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
        """
        To broadcast the communication addresses of clients (used for additive secret sharing)
        """
        self.comm_manager.send(
            Message(msg_type='address',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    content=self.comm_manager.get_neighbors()))

    def check_buffer(self, cur_round, minimal_number, check_eval_result=False):
        """
        To check the message buffer

        Arguments:
        cur_round (int): The current round number
        minimal_number (int): The minimal number of the receiving messages
        check_eval_result (bool): To check training results for evaluation results
        :returns: Whether enough messages have been received or not
        :rtype: bool
        """
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
        """
        To start the FL course when the expected number of clients have joined
        """
        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def update_best_result(self,
                           results,
                           results_type,
                           round_wise_update_key="val_loss"):
        """
            update best evaluation results.
            by default, the update is based on validation loss with `round_wise_update_key="val_loss" `
        """
        if not isinstance(results, dict):
            raise ValueError(
                f"update best results require `results` a dict, but got {type(results)}"
            )
        else:
            if results_type not in self.best_results:
                self.best_results[results_type] = dict()
            best_result = self.best_results[results_type]
            # update different keys separately: the best values can be in different rounds
            if round_wise_update_key is None:
                for key in results:
                    cur_result = results[key]
                    if 'loss' in key or 'std' in key:  # the smaller, the better
                        if results_type == "client_individual":
                            cur_result = min(cur_result)
                        if key not in best_result or cur_result < best_result[
                                key]:
                            best_result[key] = cur_result
                            logging.info(
                                f"Find new best result for {results_type}.{key} with value {cur_result}"
                            )
                    elif 'acc' in key:  # the larger, the better
                        if results_type == "client_individual":
                            cur_result = max(cur_result)
                        if key not in best_result or cur_result > best_result[
                                key]:
                            best_result[key] = cur_result
                            logging.info(
                                f"Find new best result for {results_type}.{key} with value {cur_result}"
                            )
                    else:
                        # unconcerned metric
                        pass
            # update different keys round-wise: if find better round_wise_update_key, update others at the same time
            else:
                if round_wise_update_key not in [
                        "val_loss", "val_acc", "val_std", "test_loss",
                        "test_acc", "test_std", "test_avg_loss", "loss"
                ]:
                    raise NotImplementedError(
                        f"We currently support round_wise_update_key as one of "
                        f"['val_loss', 'val_acc', 'val_std', 'test_loss', 'test_acc', 'test_std'] "
                        f"for round-wise best results update, but got {round_wise_update_key}."
                    )

                found_round_wise_update_key = False
                sorted_keys = []
                for key in results:
                    if round_wise_update_key in key:
                        sorted_keys.insert(0, key)
                        found_round_wise_update_key = True
                    else:
                        sorted_keys.append(key)
                if not found_round_wise_update_key:
                    raise ValueError(
                        "The round_wise_update_key is not in target results, "
                        "use another key or check the name. \n"
                        f"Your round_wise_update_key={round_wise_update_key}, "
                        f"the keys of results are {list(results.keys())}")

                update_best_this_round = False
                for key in sorted_keys:
                    cur_result = results[key]
                    if update_best_this_round or \
                            ('loss' in round_wise_update_key and 'loss' in key) or \
                            ('std' in round_wise_update_key and 'std' in key):
                        if results_type == "client_individual":
                            cur_result = min(cur_result)
                        if update_best_this_round or \
                                key not in best_result or cur_result < best_result[key]:
                            best_result[key] = cur_result
                            logging.info(
                                f"Find new best result for {results_type}.{key} with value {cur_result}"
                            )
                            update_best_this_round = True
                    elif update_best_this_round or \
                            'acc' in round_wise_update_key and 'acc' in key:
                        if results_type == "client_individual":
                            cur_result = max(cur_result)
                        if update_best_this_round or \
                                key not in best_result or cur_result > best_result[key]:
                            best_result[key] = cur_result
                            logging.info(
                                f"Find new best result for {results_type}.{key} with value {cur_result}"
                            )
                            update_best_this_round = True
                    else:
                        # unconcerned metric
                        pass

    def eval(self):
        """
        To conduct evaluation. When cfg.federate.make_global_eval=True, a global evaluation is conducted by the server.
        """
        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation in server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)
                formatted_eval_res = formatted_logging(
                    metrics,
                    rnd=self.state,
                    role='Global-Eval-Server #',
                    forms=self._cfg.eval.report)
                self.update_best_result(formatted_eval_res['Results_raw'],
                                        results_type="server_global_eval",
                                        round_wise_update_key=self._cfg.eval.
                                        best_res_update_round_wise_key)
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                logging.info(formatted_eval_res)
            self.check_and_save()
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
