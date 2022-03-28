import numpy as np
import torch


def index_to_mask(index, size, device='cpu'):
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data,
                            num_classes,
                            percls_trn=20,
                            val_lb=500,
                            Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
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


def dirichlet_distribution_noniid_slice(label, client_num, alpha):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid_partition/noniid_partition.py
    
    Arguments:
        label (torch.Tensor): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (torch.Tensor): number of predicted missing node.
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.size()) != 1:
        raise ValueError('Only support single-label tasks!')
    num = label.size(0)
    classes = len(torch.unique(label))
    # min number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            prop = np.array([
                p * (len(idx_j) < num / client_num)
                for p, idx_j in zip(prop, idx_slice)
            ])
            prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            min_size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
    return maxdegree