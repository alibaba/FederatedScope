import copy

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

from federatedscope.vertical_fl.dataloader.utils import batch_iter


class nnTrainer(object):
    def __init__(self, model, data, device, config, monitor):
        self.model = model
        self.data = data
        self.device = device
        self.cfg = config
        self.monitor = monitor

        self.batch_x = None
        self.batch_y = None
        self.batch_y_hat = None
        self.bottom_model = None
        self.top_model = None

        self.criterion = nn.MSELoss()

        self.grad_partition = [0] + config.vertical.output_layer

    def sample_data(self, index=None):
        if index is None:
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    def _init_for_train(self,
                        bottom_input_layer,
                        bottom_output_layer,
                        top_input_layer=None):
        self.lr = self.cfg.train.optimizer.lr
        self.dataloader = batch_iter(self.data['train'],
                                     self.cfg.dataloader.batch_size,
                                     shuffled=True)
        self._set_bottom_model(bottom_input_layer, bottom_output_layer)
        self.bottom_model_opt = optim.SGD(self.bottom_model.parameters(),
                                          lr=self.lr)
        self.bottom_model_opt.zero_grad()
        if top_input_layer:
            self._set_top_model(top_input_layer)
            self.top_model_opt = optim.SGD(self.top_model.parameters(),
                                           lr=self.lr)
            self.top_model_opt.zero_grad()

    def fetch_train_data(self, index=None):
        # Fetch new data
        self.bottom_model_opt.zero_grad()
        if self.top_model:
            self.top_model_opt.zero_grad()
        if not index:
            batch_index, self.batch_x, self.batch_y = self.sample_data(index)
            # convert 'range' to 'list'
            #   to support gRPC protocols in distributed mode
            batch_index = list(batch_index)
        else:
            self.batch_x = self.sample_data(index)
            batch_index = 'None'

        self.batch_x = torch.Tensor(self.batch_x)
        if self.batch_y is not None:
            self.batch_y = np.vstack(self.batch_y).reshape(-1, 1)
            self.batch_y = torch.Tensor(self.batch_y)

        return batch_index

    def train_bottom(self):
        self.middle_result = self.bottom_model(self.batch_x)
        middle_result = self.middle_result.data  # detach().requires_grad_()
        return middle_result

    def train_top(self, input_):

        train_loss, grad = self.protect_grad(input_)
        self.top_model_opt.step()

        grad_list = []
        for i in range(len(self.grad_partition) - 1):
            grad_list.append(
                grad[:, self.grad_partition[i]:self.grad_partition[i] +
                     self.grad_partition[i + 1]])
        my_grad = grad_list[-1]
        self.bottom_model_backward(my_grad)

        return train_loss, grad_list[:-1]

    def bottom_model_backward(self, grad=None):
        self.middle_result.backward(grad)
        self.bottom_model_opt.step()

    def _set_bottom_model(self, input_layer, out_put_layer):
        self.bottom_input_layer = input_layer
        self.bottom_output_layer = out_put_layer
        self.bottom_model = nn.Sequential(
            nn.Linear(input_layer, out_put_layer), nn.ReLU())

    def _set_top_model(self, input_layer, out_put_layer=1):
        self.top_input_layer = input_layer
        self.top_model = nn.Sequential(nn.Linear(input_layer, out_put_layer),
                                       nn.Sigmoid())

    def protect_grad(self, input_):

        fake_grad = 0
        algo = None
        para = None
        # The following protect method is proposed in
        # "Differentially Private Label Protection in Split Learning"
        if self.cfg.vertical.protect_method == 'dp':
            args = self.cfg.vertical.protect_args[0] if len(
                self.cfg.vertical.protect_args) > 0 else {}
            algo = args.get('algo', 'Laplace')
            para = args.get('para', 1)
            repeat_model = copy.deepcopy(self.top_model)
            repeat_input = input_.detach().requires_grad_()
            repeat_y_hat = repeat_model(repeat_input)
            fake_y = torch.Tensor(
                np.vstack(np.ones(len(self.batch_y))).reshape(
                    -1, 1)) - self.batch_y
            fake_train_loss = self.criterion(repeat_y_hat, fake_y)
            fake_train_loss.backward()
            fake_grad = repeat_input.grad

        y_hat = self.top_model(input_)
        train_loss = self.criterion(y_hat, self.batch_y)
        train_loss.backward()
        grad = input_.grad

        if algo == 'Laplace':
            u = np.random.laplace(para)
        elif algo == 'Bernoulli':
            u = np.random.binomial(1, para)
        else:
            u = 0

        grad = grad + u * (fake_grad - grad)

        return train_loss, grad
