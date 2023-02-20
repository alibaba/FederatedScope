import logging
import os
import os.path as osp

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class Adult(object):
    """
    Adult Data Set
    (https://archive.ics.uci.edu/ml/datasets/adult)
    Fields
    The dataset contains 15 columns
    Training set: 'adult.data', 32561 instances
    Testing set: 'adult.test', 16281 instances
    Target filed: Income
    -- The income is divide into two classes: <=50K and >50K
    Number of attributes: 14
    -- These are the demographics and other features to describe a person

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
    base_folder = 'adult'
    url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com/adult.zip'
    raw_file = ['adult.data', 'adult.test']

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
        super(Adult, self).__init__()
        self.root = root
        self.num_of_clients = num_of_clients
        self.tr_frac = tr_frac
        self.feature_partition = feature_partition
        self.seed = seed
        self.args = args
        self.algo = algo
        self.data_size_for_debug = debug_size
        self.data_dict = {}
        self.data = {}

        if download:
            self.download()
        self._get_data()

    def _get_data(self):
        fpath = os.path.join(self.root, self.base_folder)
        train_file = osp.join(fpath, 'adult.data')
        test_file = osp.join(fpath, 'adult.test')
        train_data = self._read_raw(train_file)
        test_data = self._read_raw(test_file)
        train_data, test_data = self._process(train_data, test_data)
        if self.data_size_for_debug != 0:
            subset_size = min(len(train_data), self.data_size_for_debug)
            np.random.shuffle(train_data)
            train_data = train_data[:subset_size]
        self._partition_data(train_data, test_data)

    def _read_raw(self, file_path):
        data = pd.read_csv(file_path, header=None)
        return data

    def _process(self, train_set, test_set):
        col_labels = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
            'wage_class'
        ]

        train_set.columns = col_labels
        test_set.columns = col_labels
        train_set = train_set.replace(' ?', np.nan).dropna()
        test_set = test_set.replace(' ?', np.nan).dropna()

        test_set['wage_class'] = test_set.wage_class.replace({
            ' <=50K.': ' <=50K',
            ' >50K.': ' >50K'
        })

        combined_set = pd.concat([train_set, test_set], axis=0)
        for feature in combined_set.columns:
            if combined_set[feature].dtype == 'object':
                combined_set[feature] = pd.Categorical(
                    combined_set[feature]).codes

        train_set = combined_set[:train_set.shape[0]]
        test_set = combined_set[train_set.shape[0]:]
        train_set = train_set.values
        test_set = test_set.values
        return train_set, test_set

    # normalization
    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # standardization
    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    def _partition_data(self, train_set, test_set):
        x, y = train_set[:, :-1], train_set[:, -1]
        test_x, test_y = test_set[:, :-1], test_set[:, -1]

        # change the labels from 0 to -1 to fit the 'lr' model
        if self.algo == 'lr':
            for i in range(len(y)):
                if y[i] == 0:
                    y[i] = -1
            for i in range(len(test_y)):
                if test_y[i] == 0:
                    test_y[i] = -1

        if self.args['normalization']:
            x = self.normalization(x)
            test_x = self.normalization(test_x)

        if self.args['standardization']:
            x = self.standardization(x)
            test_x = self.standardization(test_x)

        test_data = {'x': test_x, 'y': test_y}

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

    def _check_existence(self, file):
        fpath = os.path.join(self.root, self.base_folder, file)
        return osp.exists(fpath)

    def download(self):
        for file in self.raw_file:
            if self._check_existence(file):
                logger.info(file + " files already exist")
            else:
                download_and_extract_archive(self.url,
                                             os.path.join(
                                                 self.root, self.base_folder),
                                             filename=self.url.split('/')[-1])
