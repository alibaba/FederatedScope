import torch
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected
from torch_geometric.data import Data

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform

INF = np.iinfo(np.int64).max


def load_nodelevel_dataset(config=None):
    r"""
    :returns:
        data_dict
    :rtype:
        Dict: dict{'client_id': Data()}
    """
    path = config.data.root
    name = config.data.type.lower()

    # TODO: remove splitter
    # Splitter
    splitter = get_splitter(config)
    # Transforms
    transforms_funcs = get_transform(config, 'torch_geometric')
    # Dataset
    if name in ["cora", "citeseer", "pubmed"]:
        num_split = {
            'cora': [232, 542, INF],
            'citeseer': [332, 665, INF],
            'pubmed': [3943, 3943, INF],
        }
        dataset = Planetoid(path,
                            name,
                            split='random',
                            num_train_per_class=num_split[name][0],
                            num_val=num_split[name][1],
                            num_test=num_split[name][2],
                            **transforms_funcs)
        dataset = splitter(dataset[0])
        global_dataset = Planetoid(path,
                                   name,
                                   split='random',
                                   num_train_per_class=num_split[name][0],
                                   num_val=num_split[name][1],
                                   num_test=num_split[name][2],
                                   **transforms_funcs)
    elif name == "dblp_conf":
        from federatedscope.gfl.dataset.dblp_new import DBLPNew
        dataset = DBLPNew(path,
                          FL=1,
                          splits=config.data.splits,
                          **transforms_funcs)
        global_dataset = DBLPNew(path,
                                 FL=0,
                                 splits=config.data.splits,
                                 **transforms_funcs)
    elif name == "dblp_org":
        from federatedscope.gfl.dataset.dblp_new import DBLPNew
        dataset = DBLPNew(path,
                          FL=2,
                          splits=config.data.splits,
                          **transforms_funcs)
        global_dataset = DBLPNew(path,
                                 FL=0,
                                 splits=config.data.splits,
                                 **transforms_funcs)
    elif name.startswith("csbm"):
        from federatedscope.gfl.dataset.cSBM_dataset import \
            dataset_ContextualSBM
        dataset = dataset_ContextualSBM(
            root=path,
            name=name if len(name) > len("csbm") else None,
            theta=config.data.cSBM_phi,
            epsilon=3.25,
            n=2500,
            d=5,
            p=1000,
            train_percent=0.2)
        global_dataset = None
    else:
        raise ValueError(f'No dataset named: {name}!')

    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_dict = dict()
    for client_idx in range(1, len(dataset) + 1):
        local_data = dataset[client_idx - 1]
        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0])[0]
        data_dict[client_idx] = {
            'data': local_data,
            'train': [local_data],
            'val': [local_data],
            'test': [local_data]
        }
    # Keep ML split consistent with local graphs
    if global_dataset is not None:
        global_graph = global_dataset[0]
        train_mask = torch.zeros_like(global_graph.train_mask)
        val_mask = torch.zeros_like(global_graph.val_mask)
        test_mask = torch.zeros_like(global_graph.test_mask)

        for client_sampler in data_dict.values():
            if isinstance(client_sampler, Data):
                client_subgraph = client_sampler
            else:
                client_subgraph = client_sampler['data']
            train_mask[client_subgraph.index_orig[
                client_subgraph.train_mask]] = True
            val_mask[client_subgraph.index_orig[
                client_subgraph.val_mask]] = True
            test_mask[client_subgraph.index_orig[
                client_subgraph.test_mask]] = True
        global_graph.train_mask = train_mask
        global_graph.val_mask = val_mask
        global_graph.test_mask = test_mask

        data_dict[0] = {
            'data': global_graph,
            'train': [global_graph],
            'val': [global_graph],
            'test': [global_graph]
        }
    return data_dict, config
