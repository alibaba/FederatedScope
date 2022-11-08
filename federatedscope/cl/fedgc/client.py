import torch
import logging
import copy
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict

logger = logging.getLogger(__name__)


class GlobalContrastFLClient(Client):
    r"""
    GlobalContrastFL(Fedgc) Client receive aggregated model weight from
    server then update local weight; it also receive global loss from server
    to train model and update weight locally.
    """
    def _register_default_handlers(self):
        self.register_handlers('assign_client_id',
                               self.callback_funcs_for_assign_id)
        self.register_handlers('ask_for_join_in_info',
                               self.callback_funcs_for_join_in_info)
        self.register_handlers('address', self.callback_funcs_for_address)
        self.register_handlers('model_para',
                               self.callback_funcs_for_pred_embedding)
        self.register_handlers('global_loss',
                               self.callback_funcs_for_local_backward)
        self.register_handlers('ss_model_para',
                               self.callback_funcs_for_model_para)

        self.register_handlers('evaluate', self.callback_funcs_for_evaluate)
        self.register_handlers('finish', self.callback_funcs_for_finish)
        self.register_handlers('converged', self.callback_funcs_for_converged)

    def callback_funcs_for_local_backward(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        global_loss = content['global_loss']
        model_para = self.trainer.train_with_global_loss(global_loss)
        self.trainer.update(model_para)
        self.state = round
        sample_size = self.trainer.num_samples
        model_para = self.trainer.get_model_para()

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para)))

    def callback_funcs_for_pred_embedding(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.trainer.update(content)
        sample_size, model_para, results = self.trainer.train()
        self.state = round
        pred_embedding = self.trainer.get_train_pred_embedding()

        train_log_res = self._monitor.format_eval_res(results,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True)
        logger.info(train_log_res)

        self.comm_manager.send(
            Message(msg_type='pred_embedding',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(pred_embedding)))
