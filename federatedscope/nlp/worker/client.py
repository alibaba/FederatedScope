import logging
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.core.workers import Client
from federatedscope.nlp.trainer.utils import ContrastiveMonitor

logger = logging.getLogger(__name__)


class FedNLPClient(Client):
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

        self.trainer.update_stat(self.ID)

    def callback_funcs_for_model_para(self, message: Message):
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content = message.state, message.content
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should a state_dict for " \
                        "single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        else:
            round, sender, content = \
                message.state, message.sender, message.content
            self.trainer.update(content)
            self.state = round
            if self.early_stopper.early_stopped:
                sample_size, model_para_all, results = \
                    0, self.trainer.get_model_para(), {}
                logger.info(
                    f"Client #{self.ID} has been early stopped, we will "
                    f"skip the local training")
            else:
                sample_size, model_para_all, results = self.trainer.train()
                logger.info(
                    self._monitor.format_eval_res(results,
                                                  rnd=self.state + 1,
                                                  role='Client #{}'.format(
                                                      self.ID),
                                                  return_raw=True))

            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss:
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of multiple state_dict " \
                        "for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                    # print(model_para)
                    # print(self.ss_manager.secret_reconstruct(
                    # model_para_list))
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        content_frame = model_para_list_all[0][frame_idx] \
                            if single_model_case else \
                            [model_para_list[frame_idx] for model_para_list
                             in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    content=content_frame))
                        frame_idx += 1
                content_frame = model_para_list_all[0][frame_idx] if \
                    single_model_case else [
                    model_para_list[frame_idx] for model_para_list in
                    model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else:
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            content=(sample_size, model_para_all)))

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content)
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
            self.history_results = merge_dict(
                self.history_results, formatted_eval_res['Results_raw'])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))


class PFedNLPClient(FedNLPClient):
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
        self.trainer.update(content['model_para'])
        self.trainer.update_task(content['task'])
        self.state = round
        if self.early_stopper.early_stopped:
            sample_size, model_para_all, model_grads, results = \
                0, self.trainer.get_model_para(), \
                self.trainer.get_model_grads(), {}
            logger.info(f"Client #{self.ID} has been early stopped, "
                        f"we will skip the local training")
        else:
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
                        'model_grads': model_grads
                    }))

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content['model_para'])
            self.trainer.update_task(message.content['task'])
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
            self.history_results = merge_dict(
                self.history_results, formatted_eval_res['Results_raw'])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))


class PCFedNLPClient(FedNLPClient):
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

    def _copy_contrast_monitor(self, raw_monitor):
        monitor = ContrastiveMonitor()
        for var in vars(monitor):
            getattr(monitor,
                    'update_{}'.format(var))(getattr(raw_monitor, var))
        return monitor

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        last_contrast_monitor = self._copy_contrast_monitor(
            content['contrast_monitor'])
        self.state = round

        if last_contrast_monitor.stat == 1:
            self.trainer.update(content['model_para'])
        self.trainer.update_contrast_monitor(last_contrast_monitor)

        sample_size, model_para_all, model_grads, contrast_monitor, results = \
            self.trainer.train()

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
            self.trainer.update_task(message.content['task'])
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
            self.history_results = merge_dict(
                self.history_results, formatted_eval_res['Results_raw'])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
