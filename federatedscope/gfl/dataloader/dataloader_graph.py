import torch
import random
import numpy as np
import torch_geometric.transforms as transforms

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, MoleculeNet

from federatedscope.gfl.dataset.utils import get_maxDegree
from federatedscope.gfl.dataset.splitter import GraphTypeSplitter, ScaffoldSplitter, RandChunkSplitter


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def load_graphlevel_dataset(config=None):
    r"""
    Returns:
         data_local_dict (Dict): {
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
    if config.data.splitter == 'graph_type':
        alpha = 0.5
        splitter = GraphTypeSplitter(config.federate.client_num, alpha)
    elif config.data.splitter == 'scaffold':
        splitter = ScaffoldSplitter(config.federate.client_num)
    elif config.data.splitter == 'rand_chunk':
        splitter = RandChunkSplitter(config.federate.client_num)
    else:
        splitter = None

    # Transforms
    transform = transforms.Compose(eval(config.data.transform))
    pre_transform = transforms.Compose(eval(config.data.pre_transform))

    if name in [
            'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
            'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
            'REDDIT-BINARY'
    ]:
        # Add feat for datasets without attrubute
        if name in ['IMDB-BINARY', 'IMDB-MULTI'] and pre_transform is None:
            pre_transform = T.Constant(value=1.0, cat=False)
        dataset = TUDataset(path,
                            name,
                            pre_transform=pre_transform,
                            transform=transform)
        if splitter is None:
            raise ValueError('Please set the splitter.')
        dataset = splitter(dataset)

    elif name in [
            'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP',
            'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX'
    ]:
        dataset = MoleculeNet(path,
                              name,
                              pre_transform=pre_transform,
                              transform=transform)
        if splitter is None:
            raise ValueError('Please set the splitter.')
        dataset = splitter(dataset)
    elif name.startswith('graph_multi_domain'.upper()):
        if name.endswith('mol'.upper()):
            dnames = ['MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1']
        elif name.endswith('small'.upper()):
            dnames = [
                'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'ENZYMES', 'DD',
                'PROTEINS'
            ]
        elif name.endswith('mix'.upper()):
            if not pre_transform:
                raise ValueError(f'pre_transform is None!')
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
        # We provide kddcup dataset here.
        elif name.endswith('kddcupv1'.upper()):
            dnames = [
                'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                'Mutagenicity', 'NCI109', 'PTC_MM', 'PTC_FR'
            ]
        elif name.endswith('kddcupv2'.upper()):
            dnames = ['TBD']
        else:
            raise ValueError(f'No dataset named: {name}!')
        dataset = []
        # Some datasets contain x
        for dname in dnames:
            if dname.startswith('IMDB') or dname == 'COLLAB':
                tmp_dataset = TUDataset(path,
                                        dname,
                                        pre_transform=pre_transform,
                                        transform=transform)
            else:
                tmp_dataset = TUDataset(path,
                                        dname,
                                        pre_transform=None,
                                        transform=transform)
            #tmp_dataset = [ds for ds in tmp_dataset]
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
        index = np.random.permutation(np.arange(len(gs)))
        train_idx = index[:int(len(gs) * splits[0])]
        valid_idx = index[int(len(gs) *
                              splits[0]):int(len(gs) * sum(splits[:2]))]
        test_idx = index[int(len(gs) * sum(splits[:2])):]
        dataloader = {
            'num_label': get_numGraphLabels(gs),
            'train': DataLoader([gs[idx] for idx in train_idx],
                                batch_size,
                                shuffle=True,
                                num_workers=config.data.num_workers),
            'val': DataLoader([gs[idx] for idx in valid_idx],
                              batch_size,
                              shuffle=False,
                              num_workers=config.data.num_workers),
            'test': DataLoader([gs[idx] for idx in test_idx],
                               batch_size,
                               shuffle=False,
                               num_workers=config.data.num_workers),
        }
        data_local_dict[client_idx + 1] = dataloader
        raw_train = raw_train + [gs[idx] for idx in train_idx]
        raw_valid = raw_valid + [gs[idx] for idx in valid_idx]
        raw_test = raw_test + [gs[idx] for idx in test_idx]
    if not name.startswith('graph_multi_domain'.upper()):
        data_local_dict[0] = {
            'train': DataLoader(raw_train, batch_size, shuffle=True),
            'val': DataLoader(raw_valid, batch_size, shuffle=False),
            'test': DataLoader(raw_test, batch_size, shuffle=False),
        }

    return data_local_dict, config
