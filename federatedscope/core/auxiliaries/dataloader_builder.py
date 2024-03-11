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
        split: current split (default: ``train``), if split is ``test``, \
        ``cfg.dataloader.shuffle`` will be ``False``. And in PyG, ``test`` \
        split will use ``NeighborSampler`` by default.

    Returns:
        Instance of specific ``DataLoader`` configured by config.

    Note:
      The key-value pairs of ``dataloader.type`` and ``DataLoader``:
        ========================  ===============================
        ``dataloader.type``       Source
        ========================  ===============================
        ``raw``                   No DataLoader
        ``base``                  ``torch.utils.data.DataLoader``
        ``pyg``                   ``torch_geometric.loader.DataLoader``
        ``graphsaint-rw``             \
        ``torch_geometric.loader.GraphSAINTRandomWalkSampler``
        ``neighbor``              ``torch_geometric.loader.NeighborSampler``
        ``mf``                    ``federatedscope.mf.dataloader.MFDataLoader``
        ========================  ===============================
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

    if config.data.type.lower().endswith('@llm'):
        from federatedscope.llm.dataloader import get_tokenizer, \
            LLMDataCollator
        model_name, model_hub = config.model.type.split('@')
        tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                     config.llm.tok_len, model_hub)
        data_collator = LLMDataCollator(tokenizer=tokenizer)
        filtered_args['collate_fn'] = data_collator

    dataloader = loader_cls(dataset, **filtered_args)
    return dataloader
