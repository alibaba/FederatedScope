import numpy as np
import logging

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
        self.w_dict = dict()
        cfg_key_size = config.vertical.key_size
        self.public_key, self.private_key = \
            abstract_paillier.generate_paillier_keypair(n_length=cfg_key_size)
        self.vertical_dims = config.vertical.dims
        self._init_data_related_var()

        self.register_handlers('para', self.callback_func_for_para)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _init_data_related_var(self):
        self.dims = [0] + self.vertical_dims
        self.model = get_model(self._cfg.model, self.data)
        self.theta = self.model.state_dict()['fc.weight'].numpy().reshape(-1)
        self.test_x = self.data['test']['x']
        self.test_y = self.data['test']['y']

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

    def callback_func_for_para(self, message: Message):
        if message.sender == 1:
            w1, w2 = message.content
            self.w_dict[1] = [w1, w2]
        elif message.sender == 2:
            w1, w2, self.w = message.content
            self.w_dict[2] = [w1, w2]
        if len(self.w_dict) == 2:
            # print(self.w)
            h1 = self.sigmoid(
                np.matmul(self.test_x[:, :self.dims[1]], self.w_dict[1][0]))
            h2 = self.sigmoid(
                np.matmul(self.test_x[:, :self.dims[1]], self.w_dict[1][1]))
            h3 = self.sigmoid(
                np.matmul(self.test_x[:, self.dims[1]:], self.w_dict[2][0]))
            h4 = self.sigmoid(
                np.matmul(self.test_x[:, self.dims[1]:], self.w_dict[2][1]))
            h = [h1, h2, h3, h4]
            res = 0
            for i in range(4):
                res += self.w[i] * h[i]
            self.w_dict = dict()
            if self._cfg.criterion.type == 'CrossEntropyLoss':
                y_hat = self.sigmoid(res)
            else:
                y_hat = res
            loss = np.mean((self.test_y - y_hat)**2)
            y_hat = (y_hat >= 0.5).astype(np.float32)
            acc = np.sum(y_hat == self.test_y) / len(self.test_y)
            print(loss, acc)
