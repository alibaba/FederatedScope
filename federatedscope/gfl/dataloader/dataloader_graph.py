from torch_geometric import transforms
from torch_geometric.datasets import TUDataset, MoleculeNet

from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.gfl.dataset.cikm_cup import CIKMCUPDataset


def load_graphlevel_dataset(config=None):
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

    elif name in [
            'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP',
            'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX'
    ]:
        dataset = MoleculeNet(path, name, **transforms_funcs)
        return dataset, config
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
    elif name == 'CIKM':
        dataset = CIKMCUPDataset(config.data.root)
    else:
        raise ValueError(f'No dataset named: {name}!')

    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_dict = dict()
    for client_idx in range(1, len(dataset) + 1):
        data_dict[client_idx] = dataset[client_idx - 1]
    return data_dict, config
