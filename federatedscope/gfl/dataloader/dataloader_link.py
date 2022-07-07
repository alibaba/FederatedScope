import torch

from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform


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

    if config.data.loader == '':
        sampler = raw_data
    elif config.data.loader == 'graphsaint-rw':
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
    else:
        raise TypeError('Unsupported DataLoader Type {}'.format(
            config.data.loader))

    return sampler


def load_linklevel_dataset(config=None):
    r"""
    :returns:
        data_local_dict
    :rtype:
        (Dict): dict{'client_id': Data()}
    """
    path = config.data.root
    name = config.data.type.lower()

    # Splitter
    splitter = get_splitter(config)

    # Transforms
    transforms_funcs = get_transform(config, 'torch_geometric')

    if name in ['epinions', 'ciao']:
        from federatedscope.gfl.dataset.recsys import RecSys
        dataset = RecSys(path,
                         name,
                         FL=True,
                         splits=config.data.splits,
                         **transforms_funcs)
        global_dataset = RecSys(path,
                                name,
                                FL=False,
                                splits=config.data.splits,
                                **transforms_funcs)
    elif name in ['fb15k-237', 'wn18', 'fb15k', 'toy']:
        from federatedscope.gfl.dataset.kg import KG
        dataset = KG(path, name, **transforms_funcs)
        dataset = splitter(dataset[0])
        global_dataset = KG(path, name, **transforms_funcs)
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
        # Recode train & valid & test mask for global data
        global_graph = global_dataset[0]
        train_edge_mask = torch.BoolTensor([])
        valid_edge_mask = torch.BoolTensor([])
        test_edge_mask = torch.BoolTensor([])
        global_edge_index = torch.LongTensor([[], []])
        global_edge_type = torch.LongTensor([])

        for client_sampler in data_local_dict.values():
            if isinstance(client_sampler, Data):
                client_subgraph = client_sampler
            else:
                client_subgraph = client_sampler['data']
            orig_index = torch.zeros_like(client_subgraph.edge_index)
            orig_index[0] = client_subgraph.index_orig[
                client_subgraph.edge_index[0]]
            orig_index[1] = client_subgraph.index_orig[
                client_subgraph.edge_index[1]]
            train_edge_mask = torch.cat(
                (train_edge_mask, client_subgraph.train_edge_mask), dim=-1)
            valid_edge_mask = torch.cat(
                (valid_edge_mask, client_subgraph.valid_edge_mask), dim=-1)
            test_edge_mask = torch.cat(
                (test_edge_mask, client_subgraph.test_edge_mask), dim=-1)
            global_edge_index = torch.cat((global_edge_index, orig_index),
                                          dim=-1)
            global_edge_type = torch.cat(
                (global_edge_type, client_subgraph.edge_type), dim=-1)
        global_graph.train_edge_mask = train_edge_mask
        global_graph.valid_edge_mask = valid_edge_mask
        global_graph.test_edge_mask = test_edge_mask
        global_graph.edge_index = global_edge_index
        global_graph.edge_type = global_edge_type
        data_local_dict[0] = raw2loader(global_graph, config)

    return data_local_dict, config
