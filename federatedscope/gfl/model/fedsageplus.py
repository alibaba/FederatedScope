from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from federatedscope.gfl.model import SAGE_Net
"""
https://proceedings.neurips.cc//paper/2021/file/ \
34adeb8e3242824038aa65460a47c29e-Paper.pdf
Fedsageplus models from the "Subgraph Federated Learning with Missing
Neighbor Generation" (FedSage+) paper, in NeurIPS'21
Source: https://github.com/zkhku/fedsage
"""


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)

        return inputs + rand.to(inputs.device)


class FeatGenerator(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(FeatGenerator, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.dropout = dropout
        self.sample = Sampling()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 2048)
        self.fc_flat = nn.Linear(2048, self.num_pred * self.feat_shape)

    def forward(self, x):
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))

        return x


class NumPredictor(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(NumPredictor, self).__init__()
        self.reg_1 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.reg_1(x))
        return x


# Mend the graph via NeighGen
class MendGraph(nn.Module):
    def __init__(self, num_pred):
        super(MendGraph, self).__init__()
        self.num_pred = num_pred
        for param in self.parameters():
            param.requires_grad = False

    def mend_graph(self, x, edge_index, pred_degree, gen_feats):
        device = gen_feats.device
        num_node, num_feature = x.shape
        new_edges = []
        gen_feats = gen_feats.view(-1, self.num_pred, num_feature)

        if pred_degree.device.type != 'cpu':
            pred_degree = pred_degree.cpu()
        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()
        x = x.detach()
        fill_feats = torch.vstack((x, gen_feats.view(-1, num_feature)))

        for i in range(num_node):
            for j in range(min(self.num_pred, max(0, pred_degree[i]))):
                new_edges.append(
                    np.asarray([i, num_node + i * self.num_pred + j]))

        new_edges = torch.tensor(np.asarray(new_edges).reshape((-1, 2)),
                                 dtype=torch.int64).T
        new_edges = new_edges.to(device)
        if len(new_edges) > 0:
            fill_edges = torch.hstack((edge_index, new_edges))
        else:
            fill_edges = torch.clone(edge_index)
        return fill_feats, fill_edges

    def forward(self, x, edge_index, pred_missing, gen_feats):
        fill_feats, fill_edges = self.mend_graph(x, edge_index, pred_missing,
                                                 gen_feats)

        return fill_feats, fill_edges


class LocalSage_Plus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 gen_hidden,
                 dropout=0.5,
                 num_pred=5):
        super(LocalSage_Plus, self).__init__()

        self.encoder_model = SAGE_Net(in_channels=in_channels,
                                      out_channels=gen_hidden,
                                      hidden=hidden,
                                      max_depth=2,
                                      dropout=dropout)
        self.reg_model = NumPredictor(latent_dim=gen_hidden)
        self.gen = FeatGenerator(latent_dim=gen_hidden,
                                 dropout=dropout,
                                 num_pred=num_pred,
                                 feat_shape=in_channels)
        self.mend_graph = MendGraph(num_pred)

        self.classifier = SAGE_Net(in_channels=in_channels,
                                   out_channels=out_channels,
                                   hidden=hidden,
                                   max_depth=2,
                                   dropout=dropout)

    def forward(self, data):
        x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x, data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=mend_feats, edge_index=mend_edge_index))
        return degree, gen_feat, nc_pred[:data.num_nodes]

    def inference(self, impared_data, raw_data):
        x = self.encoder_model(impared_data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(raw_data.x,
                                                      raw_data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=mend_feats, edge_index=mend_edge_index))
        return degree, gen_feat, nc_pred[:raw_data.num_nodes]


class FedSage_Plus(nn.Module):
    def __init__(self, local_graph: LocalSage_Plus):
        super(FedSage_Plus, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.classifier = local_graph.classifier
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.classifier.requires_grad_(False)

    def forward(self, data):
        x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats, mend_edge_index = self.mend_graph(data.x, data.edge_index,
                                                      degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=mend_feats, edge_index=mend_edge_index))
        return degree, gen_feat, nc_pred[:data.num_nodes]
