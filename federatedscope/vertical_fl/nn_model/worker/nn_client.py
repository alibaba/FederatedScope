import numpy as np
import logging

from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.vertical_fl.dataloader.utils import batch_iter


class nnClient(Client):
    """

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

        super(nnClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)
        self.batch_x = None
        self.data = data
        self.lr = config.train.optimizer.lr
        self.batch_index = None
        self.client_num = config.federate.client_num
        self.dims = [0] + config.vertical.dims
        self.feature_num = self.dims[self.ID] - self.dims[self.ID - 1]

        self._init_data_related_var()

        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para)
        self.register_handlers('sample_data',
                               self.callback_func_for_sample_data)
        self.register_handlers('middle_result',
                               self.callback_func_for_middle_result)
        self.register_handlers('derivative', self.callback_func_for_derivative)
        self.register_handlers('continue', self.callback_funcs_for_continue)
        self.register_handlers('eval', self.callback_func_for_eval)

    def _init_data_related_var(self):
        self.own_label = ('y' in self.data['train'])
        self.dataloader = batch_iter(self.data['train'],
                                     self._cfg.dataloader.batch_size,
                                     shuffled=True)
        self.w1 = np.random.randn(self.feature_num)
        self.w2 = np.random.randn(self.feature_num)
        if self.own_label:
            self.final_w = np.random.randn(4)

    def sample_data(self, index=None):
        if index is None:
            assert self.own_label
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mse_loss(self, y, y_pred):
        return ((y - y_pred)**2).mean()

    def deriv_sigmoid(self, x):
        # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def callback_funcs_for_model_para(self, message: Message):

        if self.own_label:
            index, self.batch_x, self.batch_y = self.sample_data()
            self.batch_index = index

            self.comm_manager.send(
                Message(msg_type='sample_data',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=self.batch_index))
            self.compute_local_nn()

    def callback_func_for_sample_data(self, message: Message):
        self.batch_index = message.content
        self.batch_x = self.sample_data(index=self.batch_index)
        self.compute_local_nn()

    def compute_local_nn(self):
        self.h1 = np.matmul(self.batch_x, self.w1)
        self.h2 = np.matmul(self.batch_x, self.w2)
        if not self.own_label:
            self.comm_manager.send(
                Message(msg_type='middle_result',
                        sender=self.ID,
                        receiver=self.client_num,
                        state=self.state,
                        content=(self.h1, self.h2)))

    def callback_func_for_middle_result(self, message: Message):
        h00, h01 = message.content
        n1 = self.sigmoid(h00)
        n2 = self.sigmoid(h01)
        n3 = self.sigmoid(self.h1)
        n4 = self.sigmoid(self.h2)
        res = 0
        n = [n1, n2, n3, n4]
        for i in range(len(self.final_w)):
            res += self.final_w[i] * n[i]

        if self._cfg.criterion.type == 'CrossEntropyLoss':
            y_pred = self.sigmoid(res)
        else:
            y_pred = res
        loss = self.mse_loss(self.batch_y, y_pred)

        d_L_d_ypred = -2 * (self.batch_y - y_pred)
        if self._cfg.criterion.type == 'CrossEntropyLoss':
            d_ypred_d_final_w = self.deriv_sigmoid(res) * np.asarray(
                [n1, n2, n3, n4])
        else:
            d_ypred_d_final_w = np.asarray([n1, n2, n3, n4])

        d_ypred_d_n1 = self.deriv_sigmoid(res) * self.final_w[0]
        d_ypred_d_n2 = self.deriv_sigmoid(res) * self.final_w[1]
        d_ypred_d_n3 = self.deriv_sigmoid(res) * self.final_w[2]
        d_ypred_d_n4 = self.deriv_sigmoid(res) * self.final_w[3]

        # d_n3_d_w = np.matmul(self.deriv_sigmoid(self.h1), self.batch_x)
        # d_n4_d_w = np.matmul(self.deriv_sigmoid(self.h2), self.batch_x)

        self.final_w -= self.lr * np.matmul(d_ypred_d_final_w, d_L_d_ypred)

        self.w1 -= self.lr * np.matmul(
            self.deriv_sigmoid(self.h1) * d_ypred_d_n3 * d_L_d_ypred,
            self.batch_x)
        self.w2 -= self.lr * np.matmul(
            self.deriv_sigmoid(self.h2) * d_ypred_d_n4 * d_L_d_ypred,
            self.batch_x)

        d_L_d_n1 = d_ypred_d_n1 * d_L_d_ypred
        d_L_d_n2 = d_ypred_d_n2 * d_L_d_ypred

        self.comm_manager.send(
            Message(msg_type='derivative',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(d_L_d_n1, d_L_d_n2)))

    def callback_func_for_derivative(self, message: Message):
        d_L_d_n1, d_L_d_n2 = message.content
        # d_n1_d_w = np.matmul(self.deriv_sigmoid(self.h1), self.batch_x)
        # d_n2_d_w = np.matmul(self.deriv_sigmoid(self.h1), self.batch_x)

        self.w1 -= self.lr * np.matmul(
            self.deriv_sigmoid(self.h1) * d_L_d_n1, self.batch_x)
        self.w2 -= self.lr * np.matmul(
            self.deriv_sigmoid(self.h1) * d_L_d_n2, self.batch_x)

        self.comm_manager.send(
            Message(msg_type='continue',
                    sender=self.ID,
                    receiver=[self.client_num],
                    state=self.state,
                    content='None'))

    def callback_funcs_for_continue(self, message: Message):

        self.state += 1
        if self.state % self._cfg.eval.freq == 0 and self.state != \
                self._cfg.federate.total_round_num:
            self.comm_manager.send(
                Message(msg_type='eval',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content='None'))
            self.comm_manager.send(
                Message(msg_type='para',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=(self.w1, self.w2, self.final_w)))
            index, self.batch_x, self.batch_y = self.sample_data()
            self.batch_index = index

            self.comm_manager.send(
                Message(msg_type='sample_data',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=self.batch_index))
            self.compute_local_nn()
        elif self.state < self._cfg.federate.total_round_num:
            index, self.batch_x, self.batch_y = self.sample_data()
            self.batch_index = index

            self.comm_manager.send(
                Message(msg_type='sample_data',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=self.batch_index))
            self.compute_local_nn()
        else:
            self.comm_manager.send(
                Message(msg_type='eval',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content='None'))
            self.comm_manager.send(
                Message(msg_type='para',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=(self.w1, self.w2, self.final_w)))

    def callback_func_for_eval(self, message: Message):
        self.comm_manager.send(
            Message(msg_type='para',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=(self.w1, self.w2)))
