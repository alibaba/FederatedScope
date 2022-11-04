from federatedscope.core.data.utils import filter_dict

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def get_dataloader(dataset, config, split='train'):
    """
    Instantiate a DataLoader via config.

    Args:
        dataset: dataset from which to load the data.
        config: configs containing batch_size, shuffle, etc.
        split: current split (default: 'train'), if split is 'test', shuffle
            will be `False`. And in PyG, 'test' split will use
            `NeighborSampler` by default.

    Returns:
        dataloader: Instance of specific DataLoader configured by config.

    """
    # DataLoader builder only support torch backend now.
    if config.backend != 'torch':
        return None

    if config.dataloader.type == 'base':
        from torch.utils.data import DataLoader
        loader_cls = DataLoader
    elif config.dataloader.type == 'raw':
        # No DataLoader
        return dataset
    elif config.dataloader.type == 'pyg':
        from torch_geometric.loader import DataLoader as PyGDataLoader
        loader_cls = PyGDataLoader
    elif config.dataloader.type == 'graphsaint-rw':
        if split == 'train':
            from torch_geometric.loader import GraphSAINTRandomWalkSampler
            loader_cls = GraphSAINTRandomWalkSampler
        else:
            from torch_geometric.loader import NeighborSampler
            loader_cls = NeighborSampler
    elif config.dataloader.type == 'neighbor':
        from torch_geometric.loader import NeighborSampler
        loader_cls = NeighborSampler
    elif config.dataloader.type == 'mf':
        from federatedscope.mf.dataloader import MFDataLoader
        loader_cls = MFDataLoader
    else:
        raise ValueError(f'data.loader.type {config.data.loader.type} '
                         f'not found!')

    raw_args = dict(config.dataloader)
    if split != 'train':
        raw_args['shuffle'] = False
        raw_args['sizes'] = [-1]
        raw_args['drop_last'] = False
        # For evaluation in GFL
        if config.dataloader.type in ['graphsaint-rw', 'neighbor']:
            raw_args['batch_size'] = 4096
            dataset = dataset[0].edge_index
    else:
        if config.dataloader.type in ['graphsaint-rw']:
            # Raw graph
            dataset = dataset[0]
        elif config.dataloader.type in ['neighbor']:
            # edge_index of raw graph
            dataset = dataset[0].edge_index
    filtered_args = filter_dict(loader_cls.__init__, raw_args)
    dataloader = loader_cls(dataset, **filtered_args)
    return dataloader
