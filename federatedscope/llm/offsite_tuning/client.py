import logging

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import b64deserializer, \
    merge_dict_of_results
from federatedscope.core.auxiliaries.trainer_builder import get_trainer

logger = logging.getLogger(__name__)


class OffsiteTuningClient(Client):
    """
    Client implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(OffsiteTuningClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

    def _register_default_handlers(self):
        super(OffsiteTuningClient, self)._register_default_handlers()
        self.register_handlers('emulator_and_adapter',
                               self.callback_funcs_for_emulator_and_adapter,
                               [None])
        self.register_handlers('eval_offsite_tuning',
                               self.callback_funcs_for_evaluate_offsite,
                               'metrics')

    def callback_funcs_for_emulator_and_adapter(self, message: Message):
        logger.info(f'Client {self.ID}: Emulator and adapter received.')
        adapter_model = b64deserializer(message.content, tool='dill')
        self._model = adapter_model
        self.trainer = get_trainer(model=adapter_model,
                                   data=self.data,
                                   device=self.device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)

    def callback_funcs_for_evaluate_offsite(self, message: Message):
        """
        The handling function for receiving the request of
        evaluating offsite tuning

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        # The content contains two parts, namely, raw/plugin model
        # and emulator model

        # Extract the raw/plugin model
        raw_model = b64deserializer(message.content[0], tool='dill')
        raw_model_trainer = get_trainer(model=raw_model,
                                        data=self.data,
                                        device=self.device,
                                        config=self._cfg,
                                        only_for_eval=True,
                                        monitor=self._monitor)
        # Extract the emulator model
        if message.content[1:] is not None:
            self.trainer.update(message.content[1] if len(message.content) == 2
                                else message.content[1:],
                                strict=self._cfg.federate.share_local_model)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                emulator_eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)
                plugin_eval_matrics = raw_model_trainer.evaluate(
                    target_data_split_name=split)

                eval_metrics = {}
                for key, value in emulator_eval_metrics.items():
                    eval_metrics['emulator.' + key] = value
                for key, value in plugin_eval_matrics.items():
                    eval_metrics['plugin.' + key] = value

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            logger.info(formatted_eval_res)
            self._monitor.update_best_result(self.best_results,
                                             formatted_eval_res['Results_raw'],
                                             results_type=f"client #{self.ID}")
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))
