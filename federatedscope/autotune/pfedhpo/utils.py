import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math


class PolicyNet(nn.Module):
    def __init__(self, in_channel, num_params):
        super(PolicyNet, self).__init__()

        self.num_params = num_params
        out_channel = num_params * num_params + num_params

        self.fc_layer = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channel),
        )

    def forward(self, client_enc):
        x = self.fc_layer(client_enc)
        mean_update = x[:, :self.num_params]
        precision_component_update = x[:, self.num_params:].reshape([-1, self.num_params, self.num_params])
        return mean_update, precision_component_update


class HyperNet(nn.Module):
    def __init__(self, input_dim, num_params, n_clients):
        super(HyperNet, self).__init__()

        self.dim = input_dim
        self.PolicyNet = PolicyNet(input_dim, num_params)
        self.n_clients = n_clients

        self.alpha1 = 0.9
        self.alpha2 = 0.9
        self.T = 1.

    def forward(self, client_enc):
        mean, cov = self.PolicyNet(client_enc)
        cov_matrix = F.relu(torch.matmul(cov, cov.transpose(1, 2)))
        dist = MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)
        sample = dist.sample()
        logprob = dist.log_prob(sample)

        entropy = dist.entropy()
        param_raw = F.sigmoid(sample / self.T)

        return param_raw, logprob, entropy


def parse_pbounds(search_space):
    pbounds = {}
    for k, v in search_space.items():
        if not (hasattr(v, 'lower') and hasattr(v, 'upper')):
            raise ValueError("Unsupported hyper type {}".format(type(v)))
        else:
            if v.log:
                l, u = np.log10(v.lower), np.log10(v.upper)
            else:
                l, u = v.lower, v.upper
            pbounds[k] = (l, u)
    return pbounds


def x2conf(x, pbounds, ss):
    x = np.array(x).reshape(-1)
    assert len(x) == len(pbounds)
    params = {}

    for i, (k, b) in zip(range(len(x)), pbounds.items()):
        p_inst = ss[k]
        l, u = b
        p = float(1. * x[i] * (u - l) + l)
        if p_inst.log:
            p = 10 ** p
        params[k] = int(p) if 'int' in str(type(p_inst)).lower() else p
    return params
