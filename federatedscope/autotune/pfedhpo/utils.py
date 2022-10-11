import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math

class EncNet(nn.Module):
    def __init__(self, in_channel, num_params, hid_dim=64):
        super(EncNet, self).__init__()
        self.num_params = num_params

        self.fc_layer = nn.Sequential(
            nn.Linear(in_channel, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, num_params),
        )

        # self.fc_layer = nn.Sequential(
        #     spectral_norm( nn.Linear(in_channel, hid_dim)),
        #     nn.ReLU(inplace=True),
        #     spectral_norm(nn.Linear(hid_dim, num_params)),
        # )

    def forward(self, client_enc):
        mean_update = self.fc_layer(client_enc)
        return mean_update

class PolicyNet(nn.Module):
    def __init__(self, in_channel, num_params, hid_dim=32):
        super(PolicyNet, self).__init__()
        self.num_params = num_params

        # self.fc_layer = nn.Sequential(
        #     spectral_norm(nn.Linear(in_channel, hid_dim)),
        #     nn.ReLU(inplace=True),
        #     spectral_norm(nn.Linear(hid_dim, hid_dim)),
        #     nn.ReLU(inplace=True),
        #     spectral_norm(nn.Linear(hid_dim, num_params)),
        # )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_channel, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, num_params),
        )

    def forward(self, client_enc):
        mean_update = self.fc_layer(client_enc)
        return mean_update

class HyperNet(nn.Module):
    def __init__(self, encoding, num_params, n_clients, device, var):
        super(HyperNet, self).__init__()
        self.dim = input_dim = encoding.shape[1]
        self.var = var
        self.encoding = torch.nn.Parameter(encoding, requires_grad=True)
        self.mean = torch.zeros((n_clients, num_params)).to(device) + 0.5

        self.EncNet = EncNet(input_dim, num_params)
        self.meanNet = PolicyNet(num_params, num_params)
        self.combine = nn.Sequential(
            nn.Linear(num_params*2, num_params),
            nn.Sigmoid()
        )

        self.alpha = 0.8


    def forward(self):
        client_enc = self.EncNet(self.encoding)
        mean_update = self.meanNet(self.mean)
        mean = self.combine(torch.cat([client_enc, mean_update],
                                      dim=-1))

        cov_matrix = torch.eye(mean.shape[-1]).to(mean.device) * self.var
        dist = MultivariateNormal(loc=mean,
                    covariance_matrix=cov_matrix)
        sample = dist.sample()
        sample = torch.clamp(sample, 0., 1.)
        logprob = dist.log_prob(sample)
        entropy = dist.entropy()

        self.mean.data.copy_(mean.data)

        return sample, logprob, entropy


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
