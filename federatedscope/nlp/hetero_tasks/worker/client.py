import logging
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.core.workers import Client
from federatedscope.nlp.hetero_tasks.trainer.utils import ContrastiveMonitor

logger = logging.getLogger(__name__)


class ATCClient(Client):
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

        super().__init__(
            ID=ID,
            server_id=server_id,
            state=state,
            config=config,
            data=data,
            model=model,
            device=device,
            strategy=strategy,
            *args,
            **kwargs,
        )

        self.use_contrastive_loss = self._cfg.model.use_contrastive_loss
        self.trainer.update_stat(self.ID)

    def _copy_contrast_monitor(self, raw_monitor):
        monitor = ContrastiveMonitor()
        for var in vars(monitor):
            getattr(monitor,
                    'update_{}'.format(var))(getattr(raw_monitor, var))
        return monitor

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.state = round

        if not self.use_contrastive_loss:
            self.trainer.update(content['model_para'])
            self.trainer.update_pretrain_task(content['task'])

            sample_size, model_para_all, model_grads, results = \
                self.trainer.train()

            logger.info(
                self._monitor.format_eval_res(results,
                                              rnd=self.state + 1,
                                              role='Client #{}'.format(
                                                  self.ID),
                                              return_raw=True))
            self.comm_manager.send(
                Message(msg_type='model_para',
                        sender=self.ID,
                        receiver=[sender],
                        state=self.state,
                        content={
                            'sample_size': sample_size,
                            'model_para': model_para_all,
                            'model_grads': model_grads,
                        }))

        else:
            last_contrast_monitor = self._copy_contrast_monitor(
                content['contrast_monitor'])
            if last_contrast_monitor.stat == 1:
                self.trainer.update(content['model_para'])
            self.trainer.update_contrast_monitor(last_contrast_monitor)

            sample_size, model_para_all, model_grads, contrast_monitor, \
                results = self.trainer.train()

            if contrast_monitor.stat == 2:
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            content={'contrast_monitor': contrast_monitor}))

            elif contrast_monitor.stat == 3:
                logger.info(
                    self._monitor.format_eval_res(results,
                                                  rnd=self.state + 1,
                                                  role='Client #{}'.format(
                                                      self.ID),
                                                  return_raw=True))

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            content={
                                'sample_size': sample_size,
                                'model_para': model_para_all,
                                'model_grads': model_grads,
                                'contrast_monitor': contrast_monitor
                            }))

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content['model_para'])
            self.trainer.update_pretrain_task(message.content['task'])
        if self.early_stopper.early_stopped:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state + 1,
                                                      role='Client #{}'.format(
                                                          self.ID)))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state + 1,
                role='Client #{}'.format(self.ID),
                forms='raw',
                return_raw=True)
            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
