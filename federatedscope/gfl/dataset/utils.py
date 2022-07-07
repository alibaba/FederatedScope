import torch
from torch_geometric.utils import to_networkx


def index_to_mask(index, size, device='cpu'):
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data,
                            num_classes,
                            percls_trn=20,
                            val_lb=500,
                            Flag=0):

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:],
                                       size=data.num_nodes)
    else:
        val_index = torch.cat(
            [i[percls_trn:percls_trn + val_lb] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices],
                               dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
    return maxdegree
