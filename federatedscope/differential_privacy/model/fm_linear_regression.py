from torch.nn import Parameter
from torch.nn import Module
from torch.nn.init import kaiming_normal_

import numpy as np

import torch
import math


class FMLinearRegression(Module):
    """Implementation of Functional Mechanism for linear regression refer to
    `Functional Mechanism: Regression Analysis under Differential Privacy`
    [Jun Wang, et al.](https://arxiv.org/abs/1208.0219)

    Args:
        in_channels (int): the number of dimensions
        epsilon (int): the epsilon bound for differential privacy

    Note:
        The forward function returns the average loss directly, so that we
        don't need the criterion function for fm linear regression.
    """
    def __init__(self, in_channels, epsilon):
        super(FMLinearRegression, self).__init__()
        self.w = Parameter(torch.empty(in_channels, 1))
        kaiming_normal_(self.w, a=math.sqrt(5))

        sensitivity = float(2*(1+2*in_channels+in_channels**2))

        self.laplace = torch.distributions.laplace.Laplace(loc=0, scale=sensitivity / epsilon * np.sqrt(2))

    def forward(self, x, y):
        # J=0
        lambda0 = torch.matmul(y.t(), y)
        lambda0 += self.laplace.sample(sample_shape=lambda0.size()).to(lambda0.device)
        # J=1
        lambda1 = -2 * torch.matmul(y.t(), x)
        lambda1 += self.laplace.sample(sample_shape=lambda1.size()).to(lambda1.device)
        # J=2
        lambda2 = torch.matmul(x.t(), x)
        lambda2 += self.laplace.sample(sample_shape=lambda2.size()).to(lambda2.device)
        w2 = torch.matmul(self.w, self.w.t())

        loss_total = lambda0 + torch.sum(lambda1.t() * self.w) + torch.sum(lambda2*w2)

        pred = torch.matmul(x, self.w)

        return pred, loss_total / x.size(0)



