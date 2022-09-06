import numpy as np

from torch_geometric import transforms
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, MoleculeNet

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.core.interface.base_data import ClientData, \
    StandaloneDataDict


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def load_graphlevel_dataset(config=None, client_cfgs=None):
    r"""Convert dataset to Dataloader.
    :returns:
         data_local_dict
    :rtype: Dict {
                  'client_id': {
                      'train': DataLoader(),
                      'val': DataLoader(),
                      'test': DataLoader()
                               }
                  }
    """
    splits = config.data.splits
    path = config.data.root
    name = config.data.type.upper()
    client_num = config.federate.client_num
    batch_size = config.data.batch_size

    # Splitter
    splitter = get_splitter(config)

    # Transforms
    transforms_funcs = get_transform(config, 'torch_geometric')

    if name in [
            'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
            'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
            'REDDIT-BINARY'
    ]:
        # Add feat for datasets without attrubute
        if name in ['IMDB-BINARY', 'IMDB-MULTI'
                    ] and 'pre_transform' not in transforms_funcs:
            transforms_funcs['pre_transform'] = transforms.Constant(value=1.0,
                                                                    cat=False)
        dataset = TUDataset(path, name, **transforms_funcs)
        if splitter is None:
            raise ValueError('Please set the graph.')
        dataset = splitter(dataset)

    elif name in [
            'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP',
            'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX'
    ]:
        dataset = MoleculeNet(path, name, **transforms_funcs)
        if splitter is None:
            raise ValueError('Please set the graph.')
        dataset = splitter(dataset)
    elif name.startswith('graph_multi_domain'.upper()):
        """
            The `graph_multi_domain` datasets follows GCFL
            Federated Graph Classification over Non-IID Graphs (NeurIPS 2021)
        """
        if name.endswith('mol'.upper()):
            dnames = ['MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1']
        elif name.endswith('small'.upper()):
            dnames = [
                'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'ENZYMES', 'DD',
                'PROTEINS'
            ]
        elif name.endswith('mix'.upper()):
            if 'pre_transform' not in transforms_funcs:
                raise ValueError('pre_transform is None!')
            dnames = [
                'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY',
                'IMDB-MULTI'
            ]
        elif name.endswith('biochem'.upper()):
            dnames = [
                'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                'ENZYMES', 'DD', 'PROTEINS'
            ]
        else:
            raise ValueError(f'No dataset named: {name}!')
        dataset = []
        # Some datasets contain x
        for dname in dnames:
            if dname.startswith('IMDB') or dname == 'COLLAB':
                tmp_dataset = TUDataset(path, dname, **transforms_funcs)
            else:
                tmp_dataset = TUDataset(
                    path,
                    dname,
                    pre_transform=None,
                    transform=transforms_funcs['transform']
                    if 'transform' in transforms_funcs else None)
            dataset.append(tmp_dataset)
    else:
        raise ValueError(f'No dataset named: {name}!')

    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_local_dict = dict()

    # Build train/valid/test dataloader
    raw_train = []
    raw_valid = []
    raw_test = []
    for client_idx, gs in enumerate(dataset):
        if client_cfgs is not None:
            client_cfg = config.clone()
            client_cfg.merge_from_other_cfg(
                client_cfgs.get(f'client_{client_idx+1}'))
        else:
            client_cfg = config

        index = np.random.permutation(np.arange(len(gs)))
        train_idx = index[:int(len(gs) * splits[0])]
        valid_idx = index[int(len(gs) *
                              splits[0]):int(len(gs) * sum(splits[:2]))]
        test_idx = index[int(len(gs) * sum(splits[:2])):]
        client_data = ClientData(DataLoader,
                                 client_cfg,
                                 train=[gs[idx] for idx in train_idx],
                                 val=[gs[idx] for idx in valid_idx],
                                 test=[gs[idx] for idx in test_idx])
        client_data['num_label'] = get_numGraphLabels(gs)

        data_local_dict[client_idx + 1] = client_data
        raw_train = raw_train + [gs[idx] for idx in train_idx]
        raw_valid = raw_valid + [gs[idx] for idx in valid_idx]
        raw_test = raw_test + [gs[idx] for idx in test_idx]
    if not name.startswith('graph_multi_domain'.upper()):
        data_local_dict[0] = ClientData(DataLoader,
                                        config,
                                        train=raw_train,
                                        val=raw_valid,
                                        test=raw_test)

    return StandaloneDataDict(data_local_dict, config), config
