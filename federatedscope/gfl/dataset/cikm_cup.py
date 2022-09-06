import logging

import torch
import os
from torch_geometric.data import InMemoryDataset

logger = logging.getLogger(__name__)


class CIKMCUPDataset(InMemoryDataset):
    name = 'CIKM_CUP'

    def __init__(self, root):
        super(CIKMCUPDataset, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name)

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
        pass

    def __getitem__(self, idx):
        data = {}
        for split in ['train', 'val', 'test']:
            split_data = self._load(idx, split)
            if split_data:
                data[split] = split_data
        return data


def load_cikmcup_data(config, client_cfgs=None):
    from torch_geometric.loader import DataLoader

    # Build data
    dataset = CIKMCUPDataset(config.data.root)
    config.merge_from_list(['federate.client_num', len(dataset)])

    data_dict = {}
    # Build DataLoader dict
    for client_idx in range(1, config.federate.client_num + 1):
        logger.info(f'Loading CIKMCUP data for Client #{client_idx}.')
        dataloader_dict = {}
        tmp_dataset = []

        if client_cfgs is not None:
            client_cfg = config.clone()
            client_cfg.merge_from_other_cfg(
                client_cfgs.get(f'client_{client_idx}'))
        else:
            client_cfg = config

        if 'train' in dataset[client_idx]:
            dataloader_dict['train'] = DataLoader(
                dataset[client_idx]['train'],
                client_cfg.data.batch_size,
                shuffle=client_cfg.data.shuffle)
            tmp_dataset += dataset[client_idx]['train']
        if 'val' in dataset[client_idx]:
            dataloader_dict['val'] = DataLoader(dataset[client_idx]['val'],
                                                client_cfg.data.batch_size,
                                                shuffle=False)
            tmp_dataset += dataset[client_idx]['val']
        if 'test' in dataset[client_idx]:
            dataloader_dict['test'] = DataLoader(dataset[client_idx]['test'],
                                                 client_cfg.data.batch_size,
                                                 shuffle=False)
            tmp_dataset += dataset[client_idx]['test']
        if tmp_dataset:
            dataloader_dict['num_label'] = 0

        data_dict[client_idx] = dataloader_dict

    return data_dict, config
