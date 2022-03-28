from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from federatedscope.config import cfg
from federatedscope.register import register_model


class GAT_Net(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GAT_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GATConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GATConv(hidden, out_channels))
            else:
                self.convs.append(GATConv(hidden, hidden))
        self.dropout = dropout

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x