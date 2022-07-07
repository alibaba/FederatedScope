import numpy as np
import logging

from federatedscope.core.worker import Client
from federatedscope.core.message import Message
from federatedscope.vertical_fl.dataloader.utils import batch_iter


class vFLClient(Client):
    """
    The client class for vertical FL, which customizes the handled
    functions. Please refer to the tutorial for more details about the
    implementation algorithm
    Implementation of Vertical FL refer to `Private federated learning on
    vertically partitioned data via entity resolution and additively
    homomorphic encryption` [Hardy, et al., 2017]
    (https://arxiv.org/abs/1711.10677)
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

        super(vFLClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)
        self.data = data
        self.public_key = None
        self.theta = None
        self.batch_index = None
        self.own_label = ('y' in self.data['train'])
        self.dataloader = batch_iter(self.data['train'],
                                     self._cfg.data.batch_size,
                                     shuffled=True)

        self.register_handlers('public_keys',
                               self.callback_funcs_for_public_keys)
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para)
        self.register_handlers('encryped_gradient_u',
                               self.callback_funcs_for_encryped_gradient_u)
        self.register_handlers('encryped_gradient_v',
                               self.callback_funcs_for_encryped_gradient_v)

    def sample_data(self, index=None):
        if index is None:
            assert self.own_label
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    def callback_funcs_for_public_keys(self, message: Message):
        self.public_key = message.content

    def callback_funcs_for_model_para(self, message: Message):
        self.theta = message.content
        if self.own_label:
            index, input_x, input_y = self.sample_data()
            self.batch_index = index

            u_A = 0.25 * np.matmul(input_x, self.theta) - 0.5 * input_y
            en_u_A = [self.public_key.encrypt(x) for x in u_A]

            self.comm_manager.send(
                Message(msg_type='encryped_gradient_u',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=(self.batch_index, en_u_A)))

    def callback_funcs_for_encryped_gradient_u(self, message: Message):
        index, en_u_A = message.content
        self.batch_index = index
        input_x = self.sample_data(index=self.batch_index)
        u_B = 0.25 * np.matmul(input_x, self.theta)
        en_u_B = [self.public_key.encrypt(x) for x in u_B]
        en_u = np.expand_dims([sum(x) for x in zip(en_u_A, en_u_B)], -1)
        en_v_B = en_u * input_x

        self.comm_manager.send(
            Message(msg_type='encryped_gradient_v',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(en_u, en_v_B)))

    def callback_funcs_for_encryped_gradient_v(self, message: Message):
        en_u, en_v_B = message.content
        input_x = self.sample_data(index=self.batch_index)
        en_v_A = en_u * input_x
        en_v = np.concatenate([en_v_A, en_v_B], axis=-1)

        self.comm_manager.send(
            Message(msg_type='encryped_gradient',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=(len(input_x), en_v)))
