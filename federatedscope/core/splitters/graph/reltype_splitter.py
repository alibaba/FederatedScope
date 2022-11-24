import torch

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import BaseTransform

from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice
from federatedscope.core.splitters import BaseSplitter


class RelTypeSplitter(BaseTransform, BaseSplitter):
    """
    Split Data into small data via dirichlet distribution to \
    generate non-i.i.d data split.

    Arguments:
        client_num (int): Split data into client_num of pieces.
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5, realloc_mask=False):
        BaseSplitter.__init__(self, client_num)
        self.alpha = alpha
        self.realloc_mask = realloc_mask

    def __call__(self, data, **kwargs):
        data_list = []
        label = data.edge_type.numpy()
        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha)
        # Reallocation train/val/test mask
        train_ratio = data.train_edge_mask.sum().item() / data.num_edges
        test_ratio = data.test_edge_mask.sum().item() / data.num_edges
        for idx_j in idx_slice:
            edge_index = data.edge_index.T[idx_j].T
            edge_type = data.edge_type[idx_j]
            train_edge_mask = data.train_edge_mask[idx_j]
            valid_edge_mask = data.valid_edge_mask[idx_j]
            test_edge_mask = data.test_edge_mask[idx_j]
            if self.realloc_mask:
                num_edges = edge_index.size(-1)
                indices = torch.randperm(num_edges)
                train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
                train_edge_mask[indices[:round(train_ratio *
                                               num_edges)]] = True
                valid_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
                valid_edge_mask[
                    indices[round(train_ratio *
                                  num_edges):-round(test_ratio *
                                                    num_edges)]] = True
                test_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
                test_edge_mask[indices[-round(test_ratio * num_edges):]] = True
            sub_g = Data(x=data.x,
                         edge_index=edge_index,
                         index_orig=data.index_orig,
                         edge_type=edge_type,
                         train_edge_mask=train_edge_mask,
                         valid_edge_mask=valid_edge_mask,
                         test_edge_mask=test_edge_mask,
                         input_edge_index=to_undirected(
                             edge_index.T[train_edge_mask].T))
            data_list.append(sub_g)

        return data_list
