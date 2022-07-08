import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class SAGE_Net(torch.nn.Module):
    r"""GraphSAGE model from the "Inductive Representation Learning on
    Large Graphs" paper, in NeurIPS'17

    Source:
    https://github.com/pyg-team/pytorch_geometric/ \
    blob/master/examples/ogbn_products_sage.py

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
        super(SAGE_Net, self).__init__()

        self.num_layers = max_depth
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward_full(self, data):
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

    def forward(self, x, edge_index=None, edge_weight=None, adjs=None):
        r"""
        `train_loader` computes the k-hop neighborhood of a batch of nodes,
        and returns, for each layer, a bipartite graph object, holding the
        bipartite edges `edge_index`, the index `e_id` of the original edges,
        and the size/shape `size` of the bipartite graph.
        Target nodes are also included in the source nodes so that one can
        easily apply skip-connections or add self-loops.

        Arguments:
            x (torch.Tensor or PyG.data or tuple): node features or \
                full-batch data
            edge_index (torch.Tensor): edge index.
            edge_weight (torch.Tensor): edge weight.
            adjs (List[PyG.loader.neighbor_sampler.EdgeIndex]): \
                batched edge index
        :returns:
            x: output
        :rtype:
            torch.Tensor
        """
        if isinstance(x, torch.Tensor):
            if edge_index is None:
                for i, (edge_index, _, size) in enumerate(adjs):
                    x_target = x[:size[1]]
                    x = self.convs[i]((x, x_target), edge_index)
                    if i != self.num_layers - 1:
                        x = F.relu(x)
                        x = F.dropout(x,
                                      p=self.dropout,
                                      training=self.training)
            else:
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index, edge_weight)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[-1](x, edge_index, edge_weight)
            return x
        elif isinstance(x, Data) or isinstance(x, tuple):
            return self.forward_full(x)
        else:
            raise TypeError

    def inference(self, x_all, subgraph_loader, device):
        r"""
        Compute representations of nodes layer by layer, using *all*
        available edges. This leads to faster computation in contrast to
        immediately computing the final representations of each batch.

        Arguments:
            x_all (torch.Tensor): all node features
            subgraph_loader (PyG.dataloader): dataloader
            device (str): device
        :returns:
            x_all: output
        """
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)

        return x_all
