import torch
import os
from torch_geometric.data import InMemoryDataset, Data
from federatedscope.register import register_data
from torch_geometric.datasets import TUDataset, MoleculeNet

import numpy as np

class FSContestDataset(InMemoryDataset):
    def __init__(self, root):
        self.root = root
        super().__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'fscontest', 'processed')

    @property
    def processed_file_names(self):
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        return len([
            x for x in os.listdir(self.processed_dir)
            if not x.startswith('pre')
        ])

    def _load(self, idx, split):
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        dnames = [
            'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
            'Mutagenicity', 'NCI109', 'PTC_MM', 'PTC_FR'
        ]
        test_dnames = ['HIV', 'BACE']

        for name in dnames:
            ds = TUDataset(self.root, name)
            ds = [
                Data(edge_index=graph.edge_index, x=graph.x, y=graph.y)
                for graph in ds
            ]
            index = np.random.permutation(np.arange(len(ds)))
            train_index = index[:int(0.6 * len(index))]
            valid_index = index[int(0.6 * len(index)):int(0.8 * len(index))]
            test_index = index[int(0.8 * len(index)):]
            data = {
                'train': [ds[idx] for idx in train_index],
                'val': [ds[idx] for idx in valid_index],
                'test': [ds[idx] for idx in test_index]
            }
            data_list.append(data)

        for name in test_dnames:
            ds = MoleculeNet(self.root, name)
            ds = [
                Data(edge_index=graph.edge_index, x=graph.x, y=graph.y[0])
                for graph in ds
            ]
            index = np.random.permutation(np.arange(len(ds)))
            # DO NOT USE ALL DATA
            index = index[:1000]
            train_index = index[:int(0.6 * len(index))]
            valid_index = index[int(0.6 * len(index)):int(0.8 * len(index))]
            test_index = index[int(0.8 * len(index)):]
            data = {
                'train': [ds[idx] for idx in train_index],
                'val': [ds[idx] for idx in valid_index],
                'test': [ds[idx] for idx in test_index]
            }
            data_list.append(data)

        for idx, data in enumerate(data_list):
            os.makedirs(os.path.join(self.processed_dir, str(idx)),
                        exist_ok=True)
            torch.save(data['train'],
                       os.path.join(self.processed_dir, str(idx), 'train.pt'))
            torch.save(data['val'],
                       os.path.join(self.processed_dir, str(idx), 'val.pt'))
            torch.save(data['test'],
                       os.path.join(self.processed_dir, str(idx), 'test.pt'))

    def __getitem__(self, idx):
        data = {}
        for split in ['train', 'val', 'test']:
            split_data = self._load(idx, split)
            if split_data:
                data[split] = split_data
        return data


def load_fs_contest_data(config):
    from torch_geometric.loader import DataLoader
    from federatedscope.gfl.dataloader.dataloader_graph import get_numGraphLabels

    # Build data
    dataset = FSContestDataset(config.data.root)
    config.merge_from_list(['federate.client_num', len(dataset)])

    data_dict = {}
    # Build DataLoader dict
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {}
        tmp_dataset = []
        if 'train' in dataset[client_idx - 1]:
            dataloader_dict['train'] = DataLoader(dataset[client_idx -
                                                          1]['train'],
                                                  config.data.batch_size,
                                                  shuffle=config.data.shuffle)
            tmp_dataset += dataset[client_idx - 1]['train']
        if 'val' in dataset[client_idx - 1]:
            dataloader_dict['val'] = DataLoader(dataset[client_idx - 1]['val'],
                                                config.data.batch_size,
                                                shuffle=False)
            tmp_dataset += dataset[client_idx - 1]['val']
        if 'test' in dataset[client_idx - 1]:
            dataloader_dict['test'] = DataLoader(dataset[client_idx -
                                                         1]['test'],
                                                 config.data.batch_size,
                                                 shuffle=False)
            tmp_dataset += dataset[client_idx - 1]['test']
        if tmp_dataset:
            # TODO: specific by the task
            # dataloader_dict['num_label'] = 0 # todo: set to 0, used in gfl/model_builder.py line74
            dataloader_dict['num_label'] = get_numGraphLabels(tmp_dataset)
        data_dict[client_idx] = dataloader_dict

    return data_dict, config


def call_fs_contest_data(config, **kwargs):
    if config.data.type == "fs_contest_data":
        data, modified_config = load_fs_contest_data(config)
        return data, modified_config


register_data("fs_contest_data", call_fs_contest_data)
