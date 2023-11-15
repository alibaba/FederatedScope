import logging
import os
import os.path as osp

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class Abalone(object):
    """
    Abalone Data Set
    (https://archive.ics.uci.edu/ml/datasets/abalone)
    Data Set Information:
    Number of Instances: 4177
    Number of Attributes: 8

    Predicting the age of abalone from physical measurements.
    Given is the attribute name, attribute type, the measurement unit
        and a brief description.
    The number of rings is the value to predict:
        either as a continuous value or as a classification problem.

    Name / Data Type / Measurement Unit / Description/

    Sex / nominal / -- / M, F, and I (infant)
    Length / continuous / mm / Longest shell measurement
    Diameter / continuous / mm / perpendicular to length
    Height / continuous / mm / with meat in shell
    Whole weight / continuous / grams / whole abalone
    Shucked weight / continuous / grams / weight of meat
    Viscera weight / continuous / grams / gut weight (after bleeding)
    Shell weight / continuous / grams / after being dried
    Rings / integer / -- / +1.5 gives the age in years

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
    base_folder = 'abalone'
    url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com/abalone.zip'
    raw_file = 'abalone.data'

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
        data = self._process(data)
        if self.data_size_for_debug != 0:
            subset_size = min(len(data), self.data_size_for_debug)
            np.random.shuffle(data)
            data = data[:subset_size]
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

        self.data[self.num_of_clients]['train']['y'] = y[:]
        self.data[self.num_of_clients]['test']['y'] = test_y[:]
