import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import numpy as np
import torch as pt
from collections import namedtuple


class EncNet(nn.Module):
    def __init__(self, in_channel, num_params, hid_dim=64):
        super(EncNet, self).__init__()
        self.num_params = num_params

        # self.fc_layer = nn.Sequential(
        #     nn.Linear(in_channel, hid_dim),
        #     nn.Tanh(),
        #     nn.Linear(hid_dim, num_params),
        # )

        self.fc_layer = nn.Sequential(
            spectral_norm( nn.Linear(in_channel, hid_dim, bias=False)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(hid_dim, num_params, bias=False)),
        )

    def forward(self, client_enc):
        mean_update = self.fc_layer(client_enc)
        return mean_update

class PolicyNet(nn.Module):
    def __init__(self, in_channel, num_params, hid_dim=32):
        super(PolicyNet, self).__init__()
        self.num_params = num_params

        self.fc_layer = nn.Sequential(
            spectral_norm(nn.Linear(in_channel, hid_dim)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(hid_dim, hid_dim)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(hid_dim, num_params)),
        )

        # self.fc_layer = nn.Sequential(
        #     nn.Linear(in_channel, hid_dim),
        #     nn.Tanh(),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.Tanh(),
        #     nn.Linear(hid_dim, num_params),
        # )

    def forward(self, client_enc):
        mean_update = self.fc_layer(client_enc)
        return mean_update


class DisHyperNet(nn.Module):
    def __init__(self, encoding, cands, n_clients, device,):
        super(DisHyperNet, self).__init__()
        num_params = len(cands)
        self.dim = input_dim = encoding.shape[1]
        self.encoding = torch.nn.Parameter(encoding, requires_grad=True)
        self.EncNet = EncNet(input_dim, 64)
        loss_type = 'sphereface'
        self.enc_loss = AngularPenaltySMLoss(64, 10, loss_type)
        if loss_type == 'sphereface': self.reg_alpha = 0.1
        if loss_type == 'cosface': self.reg_alpha = 0.2
        if loss_type == 'arcface': self.reg_alpha = 0.1

        self.out = nn.ModuleList()
        for k, v in cands.items():
            self.out.append(nn.Sequential(
                nn.Linear(64, len(v), bias=False),
                nn.Softmax()
            ))

    def forward(self):
        client_enc = self.EncNet(self.encoding)
        # client_enc_reg = self.enc_loss(client_enc, torch.arange(10).to(self.encoding.device)) * self.reg_alpha
        # print('==== ', client_enc_reg)
        client_enc_reg = 0
        logits = []
        for module in self.out:
            out = module(client_enc)
            # out = torch.cat([out]*10, dim=0)
            logits.append(out)
        print(logits[0])
        return logits, client_enc_reg

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


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)



def flat_data(data, labels, device, n_labels=10, add_label=False):
  bs = data.shape[0]
  if add_label:
    gen_one_hots = pt.zeros(bs, n_labels, device=device)
    gen_one_hots.scatter_(1, labels[:, None], 1)
    labels = gen_one_hots
    return pt.cat([pt.reshape(data, (bs, -1)), labels], dim=1)
  else:
    if len(data.shape) > 2:
      return pt.reshape(data, (bs, -1))
    else:
      return data
rff_param_tuple = namedtuple('rff_params', ['w', 'b'])

def rff_sphere(x, rff_params):
  """
  this is a Pytorch version of anon's code for RFFKGauss
  Fourier transform formula from http://mathworld.wolfram.com/FourierTransformGaussian.html
  """
  w = rff_params.w
  xwt = pt.mm(x, w.t())
  z_1 = pt.cos(xwt)
  z_2 = pt.sin(xwt)
  z_cat = pt.cat((z_1, z_2), 1)
  norm_const = pt.sqrt(pt.tensor(w.shape[0]).to(pt.float32))
  z = z_cat / norm_const  # w.shape[0] == n_features / 2
  return z


def weights_sphere(d_rff, d_enc, sig, device, seed=1234):
    np.random.seed(seed)
    freq = np.random.randn(d_rff // 2, d_enc) / np.sqrt(sig)
    w_freq = pt.tensor(freq).to(pt.float32).to(device)
    return rff_param_tuple(w=w_freq, b=None)

def rff_rahimi_recht(x, rff_params):
  """
  implementation more faithful to rahimi+recht paper
  """
  w = rff_params.w
  b = rff_params.b
  xwt = pt.mm(x, w.t()) + b
  z = pt.cos(xwt)
  z = z * pt.sqrt(pt.tensor(2. / w.shape[0]).to(pt.float32))
  return z

def weights_rahimi_recht(d_rff, d_enc, sig, device):
  w_freq = pt.tensor(np.random.randn(d_rff, d_enc) / np.sqrt(sig)).to(pt.float32).to(device)
  b_freq = pt.tensor(np.random.rand(d_rff) * (2 * np.pi * sig)).to(device)
  return rff_param_tuple(w=w_freq, b=b_freq)

def data_label_embedding(data, labels, rff_params, mmd_type,
                         labels_to_one_hot=False, n_labels=None, device=None, reduce='mean'):
  assert reduce in {'mean', 'sum'}
  if labels_to_one_hot:
    batch_size = data.shape[0]
    one_hots = pt.zeros(batch_size, n_labels, device=device)
    one_hots.scatter_(1, labels[:, None], 1)
    labels = one_hots

  data_embedding = rff_sphere(data, rff_params) if mmd_type == 'sphere' else rff_rahimi_recht(data, rff_params)
  embedding = pt.einsum('ki,kj->kij', [data_embedding, labels])
  return pt.mean(embedding, 0) if reduce == 'mean' else pt.sum(embedding, 0)

def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size, mmd_type):
  assert d_rff % 2 == 0

  if mmd_type == 'sphere':
    w_freq = weights_sphere(d_rff, d_enc, rff_sigma, device)
  else:
    w_freq = weights_rahimi_recht(d_rff, d_enc, rff_sigma, device)

  def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels):
    data_emb = data_label_embedding(data_enc, labels, w_freq, mmd_type, labels_to_one_hot=True,
                                    n_labels=n_labels, device=device)
    gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)
    noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / batch_size)
    noisy_emb = data_emb + noise
    return pt.sum((noisy_emb - gen_emb) ** 2)

  return rff_mmd_loss, w_freq

def get_single_sigma_losses(train_loader, d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, mmd_type,
                            pca_vecs=None):
  assert d_rff % 2 == 0
  # w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)
  minibatch_loss, w_freq = get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, n_labels, noise_factor,
                                            train_loader.batch_size, mmd_type)

  noisy_emb = noisy_dataset_embedding(train_loader, w_freq, d_rff, device, n_labels, noise_factor, mmd_type,
                                      pca_vecs=pca_vecs)

  def single_release_loss(gen_enc, gen_labels):
    gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)
    return pt.sum((noisy_emb - gen_emb) ** 2)

  return single_release_loss, minibatch_loss, noisy_emb

def noisy_dataset_embedding(train_loader, w_freq, d_rff, device, n_labels, noise_factor, mmd_type, sum_frequency=25):
  emb_acc = []
  n_data = 0

  for data, labels in train_loader:
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=n_labels, add_label=False)
    emb_acc.append(data_label_embedding(
        data, labels, w_freq, mmd_type,
        labels_to_one_hot=True, n_labels=n_labels,
        device=device, reduce='sum'))
    # emb_acc.append(pt.sum(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), one_hots]), 0))
    n_data += data.shape[0]

    if len(emb_acc) > sum_frequency:
      emb_acc = [pt.sum(pt.stack(emb_acc), 0)]

  print('done collecting batches, n_data', n_data)
  emb_acc = pt.sum(pt.stack(emb_acc), 0) / n_data
  print(pt.norm(emb_acc), emb_acc.shape)
  noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / n_data)
  noisy_emb = emb_acc + noise
  return noisy_emb



