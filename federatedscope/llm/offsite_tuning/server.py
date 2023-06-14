import logging

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import b64serializer, \
    merge_dict_of_results
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.workers.server import Server

from federatedscope.llm.offsite_tuning.utils import \
    generate_emulator_and_adapter

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
        self.raw_model = model
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
        super(OffsiteTuningServer,
              self).__init__(ID, state, config, data, adap_model, client_num,
                             total_round_num, device, strategy, **kwargs)

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        logger.info('Server: Converting emulator and adapter...')
        emulator_and_adapter = b64serializer(self._model, tool='dill')

        self.comm_manager.send(
            Message(msg_type='emulator_and_adapter',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    timestamp=self.cur_timestamp,
                    content=emulator_and_adapter))
        trigger_train_func(**kwargs_for_trigger_train_func)

    def eval(self):
        # Update the raw model with the new adaptors
        self.raw_model.load_state_dict(self.model.state_dict(), strict=False)

        if self._cfg.federate.make_global_eval:
            # make the evaluation on raw model first
            raw_model_trainer = get_trainer(model=self.raw_model,
                                            data=self.data,
                                            device=self.device,
                                            config=self._cfg,
                                            is_attacker=self.is_attacker,
                                            monitor=self._monitor)
            raw_metrics = {}
            for split in self._cfg.eval.split:
                metrics = raw_model_trainer.evaluate(
                    target_data_split_name=split)
                raw_metrics.update(**metrics)
            for key, value in raw_metrics.items():
                raw_metrics['plugin.' + key] = value

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
                    metrics.update(**eval_metrics)
                for key, value in metrics.items():
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
            # broadcast two models for evaluation
            raw_model = b64serializer(self.raw_model, tool='dill')
            skip_broadcast = self._cfg.federate.method in ["local", "global"]
            model_para = [raw_model] + \
                         [{} if skip_broadcast else model.state_dict()
                          for model in self.models]

            rnd = self.state - 1

            self.comm_manager.send(
                Message(msg_type='eval_offsite_tuning',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=model_para))
            if self._cfg.federate.online_aggr:
                for idx in range(self.model_num):
                    self.aggregators[idx].reset()
