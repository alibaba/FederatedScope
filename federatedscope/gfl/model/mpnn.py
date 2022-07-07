import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set


class MPNNs2s(nn.Module):
    r"""MPNN from "Neural Message Passing for Quantum Chemistry" for
    regression and classification on graphs.
    Source: https://github.com/pyg-team/pytorch_geometric/blob/master
    /examples/qm9_nn_conv.py

        Arguments:
        in_channels (int): Size for the input node features.
        out_channels (int): dimension of output.
        num_nn (int): num_edge_features.
        hidden (int): Size for the output node representations. Default to 64.

    """
    def __init__(self, in_channels, out_channels, num_nn, hidden=64):
        super(MPNNs2s, self).__init__()
        self.lin0 = torch.nn.Linear(in_channels, hidden)

        nn = Sequential(Linear(num_nn, 16), ReLU(),
                        Linear(16, hidden * hidden))
        self.conv = NNConv(hidden, hidden, nn, aggr='add')
        self.gru = GRU(hidden, hidden)

        self.set2set = Set2Set(hidden, processing_steps=3, num_layers=3)
        self.lin1 = torch.nn.Linear(2 * hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        elif isinstance(data, tuple):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise TypeError('Unsupported data type!')

        self.gru.flatten_parameters()
        out = F.relu(self.lin0(x.float()))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr.float()))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out
