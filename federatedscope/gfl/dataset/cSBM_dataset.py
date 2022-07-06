#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.
"""
cSBM is a configurable random graph model for studying homophily and
heterophily.
Source: https://github.com/jianhao2016/GPRGNN

This is a script for contexual SBM model and its dataset generator.
contains functions:
        ContextualSBM
        parameterized_Lambda_and_mu
        save_data_to_pickle
    class:
        dataset_ContextualSBM

"""
import pickle
from datetime import datetime
import os
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from federatedscope.gfl.dataset.utils import random_planetoid_splits


def ContextualSBM(n, d, Lambda, p, mu, train_percent=0.01, u=None):
    """To generate a graph with specified homophilic degree, avg node
    degree, feature dimension, etc.
    Arguments:
        n (int): the number of nodes.
        d (int): the average node degree.
        Lambda (float): the parameter controlling homophilic degree.
        p (float): the dimension of node feature.
        mu (float): the mean of node feature.
        train_percent (float): (optional) the fraction of nodes used for
        training.
        u (numpy.Array): (optional) the parameter controlling the node feature.
    :returns:
        data : the constructed graph.
        u : the parameter controlling the node feature.
    :rtype:
        tuple: (PyG.Data, numpy.Array)

    """
    # n = 800 #number of nodes
    # d = 5 # average degree
    # Lambda = 1 # parameters
    # p = 1000 # feature dim
    # mu = 1 # mean of Gaussian
    # gamma = n / p

    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d) * Lambda
    y = np.ones(n)
    y[n // 2:] = -1
    y = np.asarray(y, dtype=int)

    quarter_len = n // 4
    # creating edge_index
    edge_index = [[], []]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] * y[j] > 0 and ((i // quarter_len) == (j // quarter_len)):
                if (i // quarter_len == 0) or (i // quarter_len == 2):
                    Flip = np.random.binomial(1, c_in / n)
                else:
                    Flip = np.random.binomial(1, c_out / n)
            elif (y[i] * y[j] > 0) or (i // quarter_len + j // quarter_len
                                       == 3):
                Flip = np.random.binomial(1, 0.5 * (c_in / n + c_out / n))
            else:
                if i // quarter_len == 0:
                    Flip = np.random.binomial(1, c_out / n)
                else:
                    Flip = np.random.binomial(1, c_in / n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1 / np.sqrt(p), [1, p]) if u is None else u
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu / n) * y[i] * u + Z / np.sqrt(p)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))
    # order edge list and remove duplicates if any.
    data.coalesce()

    num_class = len(np.unique(y))
    val_lb = int(n * train_percent)
    percls_trn = int(round(train_percent * n / num_class))
    data = random_planetoid_splits(data, num_class, percls_trn, val_lb)

    # add parameters to attribute
    data.Lambda = Lambda
    data.mu = mu
    data.n = n
    data.p = p
    data.d = d
    data.train_percent = train_percent

    return data, u


def parameterized_Lambda_and_mu(theta, p, n, epsilon=0.1):
    '''
    based on claim 3 in the paper,

        lambda^2 + mu^2/gamma = 1 + epsilon.

    1/gamma = p/n
    longer axis: 1
    shorter axis: 1/gamma.
    =>
        lambda = sqrt(1 + epsilon) * sin(theta * pi / 2)
        mu = sqrt(gamma * (1 + epsilon)) * cos(theta * pi / 2)
    Arguments:
        theta (float): controlling the homophilic degree.
        p (int): the dimension of node feature.
        n (int): the number of nodes.
        epsilon (float): (optional) controlling the var of node feature.
    :returns:
        Lambda : controlling the homophilic degree.
        mu : the mean of node feature.
    :rtype:
        tuple: (float, float)
    '''
    from math import pi
    gamma = n / p
    assert (theta >= -1) and (theta <= 1)
    Lambda = np.sqrt(1 + epsilon) * np.sin(theta * pi / 2)
    mu = np.sqrt(gamma * (1 + epsilon)) * np.cos(theta * pi / 2)
    return Lambda, mu


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    if file name not specified, use time stamp.
    Arguments:
        data (PyG.Data): the graph to be saved.
        p2root (str): the path of dataset folder.
        file_name (str): (optional) the name of output file.
    :returns:
        p2cSBM_data : the path of saved file.
    :returns:
        string
    '''
    now = datetime.now()
    surfix = now.strftime('%b_%d_%Y-%H:%M')
    if file_name is None:
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        tmp_data_name = file_name
    p2cSBM_data = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2cSBM_data, 'bw') as f:
        pickle.dump(data, f)
    return p2cSBM_data


class dataset_ContextualSBM(InMemoryDataset):
    r"""Create synthetic dataset based on the contextual SBM from the paper:
    https://arxiv.org/pdf/1807.09596.pdf

    Use the similar class as InMemoryDataset, but not requiring the root
    folder.

       See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Arguments:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset if not specified use time stamp.

        for {n, d, p, Lambda, mu}, with '_' as prefix: intial/feed in argument.
        without '_' as prefix: loaded from data information

        n: number nodes
        d: avg degree of nodes
        p: dimenstion of feature vector.

        Lambda, mu: parameters balancing the mixture of information,
                    if not specified, use parameterized method to generate.

        epsilon, theta: gap between boundary and chosen ellipsoid. theta is
                        angle of between the selected parameter and x-axis.
                        choosen between [0, 1] => 0 = 0, 1 = pi/2

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    #     url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self,
                 root,
                 name=None,
                 n=800,
                 d=5,
                 p=100,
                 Lambda=None,
                 mu=None,
                 epsilon=0.1,
                 theta=[-0.5, -0.25, 0.25, 0.5],
                 train_percent=0.01,
                 transform=None,
                 pre_transform=None):

        now = datetime.now()
        surfix = now.strftime('%b_%d_%Y-%H:%M').lower()
        if name is None:
            # not specifing the dataset name, create one with time stamp.
            self.name = '_'.join(['csbm_data', surfix])
        else:
            self.name = name

        self._n = n
        self._d = d
        self._p = p

        self._Lambda = Lambda
        self._mu = mu
        self._epsilon = epsilon
        self._theta = theta

        self._train_percent = train_percent

        root = osp.join(root, self.name)
        if not osp.isdir(root):
            os.makedirs(root)
        super(dataset_ContextualSBM, self).__init__(root, transform,
                                                    pre_transform)

        #         ipdb.set_trace()
        self.data, self.slices = torch.load(self.processed_paths[0])
        # overwrite the dataset attribute n, p, d, Lambda, mu
        if isinstance(self._Lambda, list):
            self.Lambda = self.data.Lambda.numpy()
            self.mu = self.data.mu.numpy()
            self.n = self.data.n.numpy()
            self.p = self.data.p.numpy()
            self.d = self.data.d.numpy()
            self.train_percent = self.data.train_percent.numpy()
        else:
            self.Lambda = self.data.Lambda.item()
            self.mu = self.data.mu.item()
            self.n = self.data.n.item()
            self.p = self.data.p.item()
            self.d = self.data.d.item()
            self.train_percent = self.data.train_percent.item()

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.raw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                if self._Lambda is None or self._mu is None:
                    # auto generate the lambda and mu parameter by angle theta.
                    self._Lambda = []
                    self._mu = []
                    for theta in self._theta:
                        Lambda, mu = parameterized_Lambda_and_mu(
                            theta, self._p, self._n, self._epsilon)
                        self._Lambda.append(Lambda)
                        self._mu.append(mu)

                if isinstance(self._Lambda, list):
                    u = None
                    for i, (Lambda,
                            mu) in enumerate(zip(self._Lambda, self._mu)):
                        tmp_data, u = ContextualSBM(self._n, self._d, Lambda,
                                                    self._p, mu,
                                                    self._train_percent, u)
                        name_split_idx = self.name.index('_', 2)
                        name = self.name[:name_split_idx] + '_{}'.format(
                            i) + self.name[name_split_idx:]
                        _ = save_data_to_pickle(tmp_data,
                                                p2root=self.raw_dir,
                                                file_name=name)

                else:
                    tmp_data, _ = ContextualSBM(self._n, self._d, self._Lambda,
                                                self._p, self._mu,
                                                self._train_percent)

                    _ = save_data_to_pickle(tmp_data,
                                            p2root=self.raw_dir,
                                            file_name=self.name)
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        if isinstance(self._Lambda, list):
            all_data = []
            for i, Lambda in enumerate(self._Lambda):
                name_split_idx = self.name.index('_', 2)
                name = self.name[:name_split_idx] + '_{}'.format(
                    i) + self.name[name_split_idx:]
                p2f = osp.join(self.raw_dir, name)
                with open(p2f, 'rb') as f:
                    data = pickle.load(f)
                all_data.append(data)
            for i in range(len(all_data)):
                all_data[i] = all_data[
                    i] if self.pre_transform is None else self.pre_transform(
                        all_data[i])
            torch.save(self.collate(all_data), self.processed_paths[0])
        else:
            p2f = osp.join(self.raw_dir, self.name)
            with open(p2f, 'rb') as f:
                data = pickle.load(f)
            data = data if self.pre_transform is None else self.pre_transform(
                data)
            torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phi', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=3.25)
    parser.add_argument('--root', default='../data/')
    parser.add_argument('--name', default='cSBM_demo')
    parser.add_argument('--num_nodes', type=int, default=800)
    parser.add_argument('--num_features', type=int, default=1000)
    parser.add_argument('--avg_degree', type=float, default=5)

    args = parser.parse_args()

    dataset_ContextualSBM(root=args.root,
                          name=args.name,
                          theta=args.phi,
                          epsilon=args.epsilon,
                          n=args.num_nodes,
                          d=args.avg_degree,
                          p=args.num_features)
