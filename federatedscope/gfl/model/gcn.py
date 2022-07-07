import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GCN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
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
