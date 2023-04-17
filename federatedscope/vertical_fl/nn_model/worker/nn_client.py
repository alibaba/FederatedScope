import numpy as np
import logging

import torch
from torch import nn, optim
from torch.autograd import Variable

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
        self.bottom_model = nn.Sequential(
            nn.Linear(self.dims[self.ID] - self.dims[self.ID - 1], 3),
            nn.ReLU())
        self.bottom_model_opt = optim.SGD(self.bottom_model.parameters(),
                                          lr=self.lr)
        self.bottom_model_opt.zero_grad()

        if self.own_label:
            self.top_model = nn.Sequential(nn.Linear(6, 1), nn.Sigmoid())
            self.top_model_opt = optim.SGD(self.top_model.parameters(),
                                           lr=self.lr)
            self.top_model_opt.zero_grad()

            self.models = [self.bottom_model, self.top_model]

            index, self.batch_x, self.batch_y = self.sample_data()
            self.batch_index = index

            self.batch_y = np.vstack(self.batch_y).reshape(-1, 1)

            self.batch_x = torch.from_numpy(self.batch_x)
            self.batch_y = torch.from_numpy(self.batch_y)

            self.batch_x = self.batch_x.to(torch.float32)
            self.batch_y = self.batch_y.to(torch.float32)

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
        self.batch_x = torch.from_numpy(self.batch_x)
        self.batch_x = self.batch_x.to(torch.float32)
        self.compute_local_nn()

    def compute_local_nn(self):

        self.a = self.bottom_model(self.batch_x)
        a = self.a.detach().requires_grad_()
        # print('----')
        # print(a)
        # print(self.a)
        if not self.own_label:
            self.comm_manager.send(
                Message(msg_type='middle_result',
                        sender=self.ID,
                        receiver=self.client_num,
                        state=self.state,
                        content=a))

    def callback_func_for_middle_result(self, message: Message):
        other_a = message.content
        # a = other_a
        a = torch.cat((other_a, self.a), 1)
        a = Variable(a, requires_grad=True)
        # print("===")
        # print(a.requires_grad_())
        # print(a)
        y_hat = self.top_model(a)
        # print(y_hat)
        # print(self.batch_y)

        criterion = nn.MSELoss()
        loss = criterion(y_hat, self.batch_y)
        # print(loss)
        loss.backward()

        other_grad = a.grad[:, :3]
        my_grad = a.grad[:, 3:]
        # print(a.grad)
        # print(other_grad)

        self.a.backward(my_grad)
        self.bottom_model_opt.step()

        self.comm_manager.send(
            Message(msg_type='derivative',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=other_grad))

        # print(self.top_model_opt)
        self.top_model_opt.step()
        # print(self.top_model_opt)

    def callback_func_for_derivative(self, message: Message):
        grad = message.content
        # print("==========")
        # print(grad)
        # # grad = torch.from_numpy(grad)
        # print('-----')

        # print(self.a)
        # print(grad)
        # input()
        self.a.backward(grad)
        # print(self.bottom_model[0].weight.grad)
        self.bottom_model_opt.step()

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
                Message(msg_type='model',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=self.models))
            index, self.batch_x, self.batch_y = self.sample_data()
            self.batch_index = index

            self.batch_y = np.vstack(self.batch_y).reshape(-1, 1)

            self.batch_x = torch.from_numpy(self.batch_x)
            self.batch_y = torch.from_numpy(self.batch_y)

            self.batch_x = self.batch_x.to(torch.float32)
            self.batch_y = self.batch_y.to(torch.float32)

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

            self.batch_y = np.vstack(self.batch_y).reshape(-1, 1)

            self.batch_x = torch.from_numpy(self.batch_x)
            self.batch_y = torch.from_numpy(self.batch_y)

            self.batch_x = self.batch_x.to(torch.float32)
            self.batch_y = self.batch_y.to(torch.float32)

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
                Message(msg_type='model',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=self.models))

    def callback_func_for_eval(self, message: Message):
        self.comm_manager.send(
            Message(msg_type='model',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=self.bottom_model))
