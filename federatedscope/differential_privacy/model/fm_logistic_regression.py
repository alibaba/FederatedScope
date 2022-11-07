from torch.nn import Parameter
from torch.nn import Module
from torch.nn.init import kaiming_normal_

import numpy as np

import torch


class FMLogisticRegression(Module):
    """Implementation of Functional Mechanism for logistic regression refer to
        `Functional Mechanism: Regression Analysis under Differential Privacy`
        [Jun Wang, et al.](https://arxiv.org/abs/1208.0219)

        Args:
            in_channels (int): the number of dimensions
            epsilon (int): the epsilon bound for differential privacy

        Note:
            The forward function returns the average loss directly, so that we
            don't need the criterion function for fm logistic regression.
    """
    def __init__(self, in_channels, epsilon):
        super(FMLogisticRegression, self).__init__()
        self.w = Parameter(torch.empty(in_channels, 1))
        kaiming_normal_(self.w)

        sensitivity = 0.25 * in_channels ** 2 + 3 * in_channels
        self.laplace = torch.distributions.laplace.Laplace(loc=0, scale=sensitivity / epsilon * np.sqrt(2))

    def forward(self, x, y):
        if len(y.size()) == 1:
            y = torch.unsqueeze(y, dim=-1)
        # J=0
        lambda0 = np.log(2)
        lambda0 += self.laplace.sample(sample_shape=[1]).to(x.device)
        # J=1
        lambda1 = 0.5 * x - y * x
        lambda1 += self.laplace.sample(sample_shape=lambda1.size()).to(lambda1.device)
        # J=2
        lambda2 = torch.matmul(x.t(), x)
        lambda2 += self.laplace.sample(sample_shape=lambda2.size()).to(lambda2.device)
        w2 = torch.matmul(self.w, self.w.t())

        loss_total = lambda0 * x.size(0) + torch.sum(lambda1.t() * self.w) + 0.125 * torch.sum(lambda2 * w2)

        pred = torch.matmul(x, self.w)

        return pred, loss_total / x.size(0)
