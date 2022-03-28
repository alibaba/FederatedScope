import os
import torch
import os.path as osp

from torch.utils.data import Dataset
from torch_geometric.data import download_url, extract_zip

LEAF_NAMES = [
    'femnist', 'celeba', 'synthetic', 'shakespeare', 'twitter', 'subreddit'
]


def is_exists(path, names):
    exists_list = [osp.exists(osp.join(path, name)) for name in names]
    return False not in exists_list


class LEAF(Dataset):
    def __init__(self, root, name, transform, target_transform):
        self.root = root
        self.name = name
        if name not in LEAF_NAMES:
            raise ValueError(f'No leaf dataset named {self.name}')
        self.transform = transform
        self.target_transform = target_transform
        self.process_file()

    @property
    def raw_file_names(self):
        names = ['all_data.zip']
        return names

    @property
    def extracted_file_names(self):
        names = ['all_data']
        return names

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def download(self):
        raise NotImplementedError

    def extract(self):
        for name in self.raw_file_names:
            extract_zip(osp.join(self.raw_dir, name), self.raw_dir)

    def process_file(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        if len(os.listdir(self.processed_dir)) == 0:
            if not is_exists(self.raw_dir, self.extracted_file_names):
                if not is_exists(self.raw_dir, self.raw_file_names):
                    self.download()
                self.extract()
            self.process()

    def process(self):
        raise NotImplementedError