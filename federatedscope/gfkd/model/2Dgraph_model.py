import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from federatedscope.gfl.model.graph_level import GNN_Net_Graph


class GNN_Net(GNN_Net_Graph):
    r"""GNN model with pre-linear layer, pooling layer
        and output layer for graph classification tasks.

    Arguments:
        in_channels (int): input channels.
        out_channels (int): output channels.
        hidden (int): hidden dim for all modules.
        max_depth (int): number of layers for gnn.
        dropout (float): dropout probability.
        gnn (str): name of gnn type, use ("gcn" or "gin").
        pooling (str): pooling method, use ("add", "mean" or "max").
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 gnn='gcn',
                 pooling='add',
                 conformer=False):
        super(GNN_Net, self).__init__()
        self.conformer = conformer

    def forward(self, data):
        if self.conformer == False:
            if isinstance(data, Batch):
                x, edge_index, batch = data.x, data.edge_index, data.batch
            elif isinstance(data, tuple):
                x, edge_index, batch = data
            else:
                raise TypeError('Unsupported data type!')
        else:
            if isinstance(data, Batch):
                x, edge_index, batch, pos = data.x, data.edge_index, data.batch
            elif isinstance(data, tuple):
                x, edge_index, batch = data
            else:
                raise TypeError('Unsupported data type!')

        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)

        x = self.gnn((x, edge_index))
        x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x