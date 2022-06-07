import torch
import os
from torch_geometric.data import InMemoryDataset
from federatedscope.register import register_data


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


def call_fs_contest_data(config):
    if config.data.type == "fs_contest_data":
        data, modified_config = load_fs_contest_data(config)
        return data, modified_config


register_data("fs_contest_data", call_fs_contest_data)
