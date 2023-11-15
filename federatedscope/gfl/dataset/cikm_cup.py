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
