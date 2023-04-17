import numpy as np
import logging

import torch

from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.vertical_fl.Paillier import abstract_paillier
from federatedscope.core.auxiliaries.model_builder import get_model

logger = logging.getLogger(__name__)


class nnServer(Server):
    """

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
        super(nnServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        self.model_dict = dict()
        cfg_key_size = config.vertical.key_size
        self.public_key, self.private_key = \
            abstract_paillier.generate_paillier_keypair(n_length=cfg_key_size)
        self.vertical_dims = config.vertical.dims
        self._init_data_related_var()

        self.register_handlers('model', self.callback_func_for_model)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _init_data_related_var(self):
        self.dims = [0] + self.vertical_dims
        self.model = get_model(self._cfg.model, self.data)
        self.theta = self.model.state_dict()['fc.weight'].numpy().reshape(-1)
        self.test_x = self.data['test']['x']
        self.test_y = self.data['test']['y']

        self.test_x = self.test_x[:, :]
        self.test_x = torch.from_numpy(self.test_x)
        self.test_x = self.test_x.to(torch.float32)

        self.test_y = np.vstack(self.test_y).reshape(-1, 1)
        self.test_y = torch.from_numpy(self.test_y)
        self.test_y = self.test_y.to(torch.float32)

    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_client_address()
            self.trigger_for_feat_engr(self.broadcast_model_para)

    def broadcast_model_para(self):
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[each for each in self.comm_manager.neighbors],
                    state=self.state,
                    content='None'))

    def callback_func_for_model(self, message: Message):
        self.model_dict[message.sender] = message.content

        if len(self.model_dict) == 2:
            # print(self.dims)
            # print(self.test_x[:, :self.dims[1]])
            a1 = self.model_dict[1](self.test_x[:, :self.dims[1]])

            a2 = self.model_dict[2][0](self.test_x[:, self.dims[1]:])
            a = torch.cat((a1, a2), 1)
            y_hat = self.model_dict[2][1](a)
            self.model_dict = dict()
            # print(y_hat)
            # print(self.test_y)
            loss = torch.mean((self.test_y - y_hat)**2)
            y_hat = (y_hat >= 0.5)
            acc = torch.sum(y_hat == self.test_y) / len(self.test_y)
            print(loss, acc)
