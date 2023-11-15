import logging
import os
import os.path as osp

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class Credit(object):
    """
    Give Me Some Credit Data Set
    (https://www.kaggle.com/competitions/GiveMeSomeCredit)
    Data Set: cs-training.csv, 150000 instances and 12 attributes
    The first attribute is the user ID which we do not need, the second
    attribute is the label, determining whether a loan should be granted.

    Arguments:
        root (str): root path
        num_of_clients(int): number of clients
        feature_partition(list): the number of features
                                    partitioned to each client
        tr_frac (float): train set proportion for each task; default=0.8
        args (dict): set Ture or False to decide whether
                     to normalize or standardize the data or not,
                     e.g., {'normalization': False, 'standardization': False}
        algo(str): the running model, 'lr'/'xgb'/'gbdt'/'rf'
        debug_size(int): use a subset for debug,
                                  0 for using entire dataset
        download (bool): indicator to download dataset
        seed: a random seed
    """
    base_folder = 'givemesomecredit'
    url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com/cs-training.zip'
    raw_file = 'cs-training.csv'

    def __init__(self,
                 root,
                 num_of_clients,
                 feature_partition,
                 args,
                 algo=None,
                 tr_frac=0.8,
                 debug_size=0,
                 download=True,
                 seed=123):
        super(Credit, self).__init__()
        self.root = root
        self.num_of_clients = num_of_clients
        self.feature_partition = feature_partition
        self.tr_frac = tr_frac
        self.seed = seed
        self.args = args
        self.algo = algo
        self.data_size_for_debug = debug_size
        self.data_dict = {}
        self.data = {}

        if download:
            self.download()
        if not self._check_existence():
            raise RuntimeError("Dataset not found or corrupted." +
                               "You can use download=True to download it")

        self._get_data()
        self._partition_data()

    def _get_data(self):
        fpath = os.path.join(self.root, self.base_folder)
        file = osp.join(fpath, self.raw_file)
        data = self._read_raw(file)
        data = data[:, 1:]

        # the following codes are used to choose balanced data
        # they may be removed later
        # '''
        sample_size = 150000

        def balance_sample(sample_size, y):
            y_ones_idx = (y == 1).nonzero()[0]
            y_ones_idx = np.random.choice(y_ones_idx,
                                          size=int(sample_size / 2))
            y_zeros_idx = (y == 0).nonzero()[0]
            y_zeros_idx = np.random.choice(y_zeros_idx,
                                           size=int(sample_size / 2))

            y_index = np.concatenate([y_zeros_idx, y_ones_idx], axis=0)
            np.random.shuffle(y_index)
            return y_index

        sample_idx = balance_sample(sample_size, data[:, 0])
        data = data[sample_idx]
        # '''

        if self.data_size_for_debug != 0:
            subset_size = min(len(data), self.data_size_for_debug)
            np.random.shuffle(data)
            data = data[:subset_size]

        train_num = int(self.tr_frac * len(data))

        self.data_dict['train'] = data[:train_num]
        self.data_dict['test'] = data[train_num:]

    def _read_raw(self, file_path):
        data = pd.read_csv(file_path)
        data = data.fillna(method='ffill')
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

        x = self.data_dict['train'][:, 1:]
        y = self.data_dict['train'][:, 0]

        test_data = {
            'x': self.data_dict['test'][:, 1:],
            'y': self.data_dict['test'][:, 0]
        }
        test_x = test_data['x']
        test_y = test_data['y']

        self.data = dict()
        for i in range(self.num_of_clients + 1):
            self.data[i] = dict()
            if i == 0:
                self.data[0]['train'] = None
                self.data[0]['test'] = test_data
            elif i == 1:
                self.data[1]['train'] = {'x': x[:, :self.feature_partition[0]]}
                self.data[1]['test'] = {
                    'x': test_x[:, :self.feature_partition[0]]
                }
            else:
                self.data[i]['train'] = {
                    'x': x[:,
                           self.feature_partition[i -
                                                  2]:self.feature_partition[i -
                                                                            1]]
                }
                self.data[i]['test'] = {
                    'x': test_x[:, self.feature_partition[i - 2]:self.
                                feature_partition[i - 1]]
                }
            self.data[i]['val'] = None

        self.data[self.num_of_clients]['train']['y'] = y
        self.data[self.num_of_clients]['test']['y'] = test_y[:]
