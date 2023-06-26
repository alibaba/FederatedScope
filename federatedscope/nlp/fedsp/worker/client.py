import logging
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.core.workers import Client

logger = logging.getLogger(__name__)


class FedSPClient(Client):
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

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.state = round
        self.trainer.update(content['model_para'])

        if self._cfg.federate.skip_local_train:
            sample_size = 1
            model_para_all = self.trainer.get_model_para()
            model_grads = self.trainer.get_model_grads()
        else:
            if self._cfg.federate.pl_alter_train:
                self.trainer.update_alter_stage('model')
                self.trainer.train()
                self.trainer.update_alter_stage('prompt')

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

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content['model_para'])
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
