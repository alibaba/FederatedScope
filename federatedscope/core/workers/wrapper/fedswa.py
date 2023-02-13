import types
import logging

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results

logger = logging.getLogger(__name__)


def wrap_swa_server(server):
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
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()

                self.state += 1

                # FedSWA cache model
                if self.state == self._cfg.fedswa.start_rnd:
                    self.swa_models_ws = [
                        model.state_dict() for model in self.models
                    ]
                    self.swa_rnd = 1
                elif self.state > \
                    self._cfg.fedswa.start_rnd and \
                        (self.state - self._cfg.fedswa.start_rnd) % \
                        self._cfg.fedswa.freq == 0:
                    logger.info(f'FedSWA cache {self.swa_rnd} models.')
                    for model, new_model in zip(self.swa_models_ws,
                                                self.models):
                        new_model = new_model.state_dict()
                        for key in model.keys():
                            model[key] = (model[key] * self.swa_rnd +
                                          new_model[key]) / (self.swa_rnd + 1)
                    self.swa_rnd += 1

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

        else:
            move_on_flag = False

        return move_on_flag

    def eval(self):
        if self._cfg.federate.make_global_eval:
            for i in range(self.model_num):
                trainer = self.trainers[i]

                if self.eval_swa:
                    # Use swa model
                    fedavg_model_w = self.models[i].state_dict()
                    self.models[i].load_state_dict(self.swa_models_ws[i])

                # Preform evaluation in server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)

                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state,
                    role='Server SWA#' if self.eval_swa else 'Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)

                if self.eval_swa:
                    # Restore
                    self.models[i].load_state_dict(fedavg_model_w)
                    self.best_results = formatted_eval_res['Results_raw']
                else:
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
            if self.eval_swa:
                for i in range(self.model_num):
                    # Use swa model
                    fedavg_model_w = self.models[i].state_dict()
                    self.models[i].load_state_dict(self.swa_models_ws[i])
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate',
                                      filter_unseen_clients=False)

            if self.eval_swa:
                for i in range(self.model_num):
                    self.models[i].load_state_dict(fedavg_model_w)

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

        if should_stop or self.state >= self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round or early stopped
            self.save_best_results()
            if not self._cfg.federate.make_global_eval:
                self.save_client_eval_results()

            if self.eval_swa:
                self.terminate(msg_type='finish')
            else:
                self.eval_swa = True
                logger.info('Server: Evaluation with FedSWA')
                self.eval()

        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval:
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear()

        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1

    def save_best_results(self):
        """
        To Save the best evaluation results.
        """
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)
        formatted_best_res = self._monitor.format_eval_res(
            results=self.best_results,
            rnd="Final",
            role='Server SWA#' if self.eval_swa else 'Server #',
            forms=["raw"],
            return_raw=True)
        logger.info(formatted_best_res)
        self._monitor.save_formatted_results(formatted_best_res)

    # Bind method to instance
    setattr(server, 'eval_swa', False)
    server.check_and_move_on = types.MethodType(check_and_move_on, server)
    server.eval = types.MethodType(eval, server)
    server.check_and_save = types.MethodType(check_and_save, server)
    server.save_best_results = types.MethodType(save_best_results, server)

    return server
