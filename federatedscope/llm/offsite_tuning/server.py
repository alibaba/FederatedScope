import os
import logging

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import b64serializer, \
    merge_dict_of_results
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.workers.server import Server

from federatedscope.llm.offsite_tuning.utils import \
    generate_emulator_and_adapter, align_student_with_teacher

logger = logging.getLogger(__name__)


class OffsiteTuningServer(Server):
    """
    Server implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
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
        compress_strategy = config.llm.offsite_tuning.strategy
        emulator_l = config.llm.offsite_tuning.emu_l
        emulator_r = config.llm.offsite_tuning.emu_r
        offsite_tuning_kwargs = config.llm.offsite_tuning.kwargs[0]
        logger.info('Server: Generating emulator and adapter...')
        adap_model = \
            generate_emulator_and_adapter(model,
                                          strategy=compress_strategy,
                                          emulator_l=emulator_l,
                                          emulator_r=emulator_r,
                                          **offsite_tuning_kwargs)
        # Emulator alignment
        if config.llm.offsite_tuning.emu_align.use:
            adap_model = align_student_with_teacher(raw_model=model,
                                                    adap_model=adap_model,
                                                    cfg=config,
                                                    device=device,
                                                    monitor=Monitor(
                                                        config,
                                                        monitored_object=self))
            if config.llm.offsite_tuning.emu_align.exit_after_align:
                os._exit(0)
        # No need for this attr
        if hasattr(adap_model, 'teacher'):
            del adap_model.teacher

        self.raw_model = model
        super(OffsiteTuningServer,
              self).__init__(ID, state, config, data, adap_model, client_num,
                             total_round_num, device, strategy, **kwargs)
        if self._cfg.llm.offsite_tuning.eval_type == 'full':
            self.raw_model_trainer = get_trainer(model=self.raw_model,
                                                 data=self.data,
                                                 device=self.device,
                                                 config=self._cfg,
                                                 only_for_eval=True,
                                                 monitor=Monitor(
                                                     self._cfg,
                                                     monitored_object=self))

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        logger.info('Server: Converting emulator and adapter...')
        if self._cfg.federate.mode == 'standalone' and \
                self._cfg.federate.share_local_model:
            logger.info('Server: `share_local_model` mode enabled, '
                        'emulator_and_adapter is built in FedRunner.')
            self.comm_manager.send(
                Message(msg_type='emulator_and_adapter',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        timestamp=self.cur_timestamp,
                        content=None))
        else:
            emulator_and_adapter = b64serializer(self._model, tool='dill')

            self.comm_manager.send(
                Message(msg_type='emulator_and_adapter',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        timestamp=self.cur_timestamp,
                        content=emulator_and_adapter))

        trigger_train_func(**kwargs_for_trigger_train_func)

    def eval(self):
        # Update the raw model with the new adapters
        if self._cfg.llm.offsite_tuning.eval_type == 'full':
            self.model.to('cpu')
            new_raw_model_state_dict = self.raw_model.state_dict()
            for key, value in zip(self.raw_model.state_dict().keys(),
                                  self.model.state_dict().values()):
                new_raw_model_state_dict[key] = value
            self.raw_model_trainer.update(new_raw_model_state_dict,
                                          strict=False)
            # make the evaluation on raw model at the server first
            raw_metrics = {}
            for split in self._cfg.eval.split:
                metrics = self.raw_model_trainer.evaluate(
                    target_data_split_name=split)
                for key, value in metrics.items():
                    raw_metrics['plugin.' + key] = value
            # Move to cpu
            self.raw_model.to('cpu')

        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation for emulator at server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    for key, value in eval_metrics.items():
                        metrics['emulator.' + key] = value
                metrics.update(**raw_metrics)
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
            super().eval()
            if self._cfg.llm.offsite_tuning.eval_type == 'full':
                self.raw_metrics = raw_metrics

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

        # The content received from the clients is the result of emulator
        self.msg_buffer['eval'][rnd][sender] = {
            'emulator.' + key: value
            for key, value in content.items()
        }
        if self._cfg.llm.offsite_tuning.eval_type == 'full':
            self.msg_buffer['eval'][rnd][sender].update(**self.raw_metrics)

        return self.check_and_move_on(check_eval_result=True)
