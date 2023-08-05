import torch
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform

INF = np.iinfo(np.int64).max


def raw2loader(raw_data, config=None):
    """Transform a graph into either dataloader for graph-sampling-based
    mini-batch training
    or still a graph for full-batch training.
    Arguments:
        raw_data (PyG.Data): a raw graph.
    :returns:
        sampler (object): a Dict containing loader and subgraph_sampler or
        still a PyG.Data object.
    """
    # change directed graph to undirected
    raw_data.edge_index = to_undirected(
        remove_self_loops(raw_data.edge_index)[0])

    if config.data.loader == '':
        sampler = raw_data
    elif config.data.loader == 'graphsaint-rw':
        # Sampler would crash if there was isolated node.
        raw_data.edge_index = add_self_loops(raw_data.edge_index,
                                             num_nodes=raw_data.x.shape[0])[0]
        loader = GraphSAINTRandomWalkSampler(
            raw_data,
            batch_size=config.data.batch_size,
            walk_length=config.data.graphsaint.walk_length,
            num_steps=config.data.graphsaint.num_steps,
            sample_coverage=0)
        subgraph_sampler = NeighborSampler(raw_data.edge_index,
                                           sizes=[-1],
                                           batch_size=4096,
                                           shuffle=False,
                                           num_workers=config.data.num_workers)
        sampler = dict(data=raw_data,
                       train=loader,
                       val=subgraph_sampler,
                       test=subgraph_sampler)
    elif config.data.loader == 'neighbor':
        # Sampler would crash if there was isolated node.
        raw_data.edge_index = add_self_loops(raw_data.edge_index,
                                             num_nodes=raw_data.x.shape[0])[0]

        train_idx = raw_data.train_mask.nonzero(as_tuple=True)[0]
        loader = NeighborSampler(raw_data.edge_index,
                                 node_idx=train_idx,
                                 sizes=config.data.sizes,
                                 batch_size=config.data.batch_size,
                                 shuffle=config.data.shuffle,
                                 num_workers=config.data.num_workers)
        subgraph_sampler = NeighborSampler(raw_data.edge_index,
                                           sizes=[-1],
                                           batch_size=4096,
                                           shuffle=False,
                                           num_workers=config.data.num_workers)
        sampler = dict(data=raw_data,
                       train=loader,
                       val=subgraph_sampler,
                       test=subgraph_sampler)

    return sampler


def load_nodelevel_dataset(config=None):
    r"""
    :returns:
        data_local_dict
    :rtype:
        Dict: dict{'client_id': Data()}
    """
    path = config.data.root
    name = config.data.type.lower()

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
    data_local_dict = dict()

    for client_idx in range(len(dataset)):
        local_data = raw2loader(dataset[client_idx], config)
        data_local_dict[client_idx + 1] = local_data

    if global_dataset is not None:
        global_graph = global_dataset[0]
        train_mask = torch.zeros_like(global_graph.train_mask)
        val_mask = torch.zeros_like(global_graph.val_mask)
        test_mask = torch.zeros_like(global_graph.test_mask)

        for client_sampler in data_local_dict.values():
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

        data_local_dict[0] = raw2loader(global_graph, config)

    return data_local_dict, config
