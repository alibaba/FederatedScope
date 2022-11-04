import logging
import os
import os.path as osp

import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class Abalone:
    def __init__(self,
                 root,
                 name,
                 num_of_clients,
                 feature_partition,
                 s_frac=0.0,
                 tr_frac=0.8,
                 val_frac=0.0,
                 train_tasks_frac=1.0,
                 download=True,
                 seed=123):
        self.root = root
        self.name = name
        self.num_of_clients = num_of_clients
        self.feature_partition = feature_partition
        self.s_frac = s_frac
        self.tr_frac = tr_frac
        self.val_frac = val_frac
        self.train_tasks_frac = train_tasks_frac
        self.seed = seed
        self.data_dict = {}
        self.data = {}

        if download:
            self.download()
        if not self._check_existence():
            raise RuntimeError("Dataset not found or corrupted." +
                               "You can use download=True to download it")

        self._get_data()
        self._partition_data()

    base_folder = 'abalone'
    url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com/abalone.zip'
    raw_file = 'abalone.data'

    def _get_data(self):
        fpath = os.path.join(self.root, self.base_folder)
        file = osp.join(fpath, self.raw_file)
        data = self._read_raw(file)
        data = self._process(data)
        train_num = int(self.tr_frac * len(data))
        self.data_dict['train'] = data[:train_num]
        self.data_dict['test'] = data[train_num:]

    def _read_raw(self, file_path):
        data = pd.read_csv(file_path, header=None)
        return data

    def _process(self, data):
        data[0] = data[0].replace({'F': 2, 'M': 1, 'I': 0})
        data = data.values
        return data

    def _check_existence(self):
        fpath = os.path.join(self.root, self.base_folder, self.raw_file)
        return osp.exists(fpath)

    def download(self):
        if self._check_existence():
            logger.info("Files already exist")
            return
        download_and_extract_archive(self.url,
                                     os.path.join(self.root, self.base_folder),
                                     filename=self.url.split('/')[-1])

    def _partition_data(self):

        x = self.data_dict['train'][:, :-1]
        y = self.data_dict['train'][:, -1]

        test_data = {
            'x': self.data_dict['test'][:, :-1],
            'y': self.data_dict['test'][:, -1]
        }

        self.data = dict()
        for i in range(self.num_of_clients + 1):
            self.data[i] = dict()
            if i == 0:
                self.data[0]['train'] = None
            elif i == 1:
                self.data[1]['train'] = {'x': x[:, :self.feature_partition[0]]}
            else:
                self.data[i]['train'] = {
                    'x': x[:,
                           self.feature_partition[i -
                                                  2]:self.feature_partition[i -
                                                                            1]]
                }
            self.data[i]['val'] = None
            self.data[i]['test'] = test_data

        self.data[self.num_of_clients]['train']['y'] = y[:]
