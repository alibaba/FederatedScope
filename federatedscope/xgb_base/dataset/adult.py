import zipfile
import os
import logging
import os.path as osp
import pandas as pd
import glob
import collections
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class Adult:
    def __init__(self,
                 root,
                 name,
                 num_of_clients,
                 feature_partition,
                 tr_frac=0.9,
                 download=True,
                 seed=123):
        super(Adult, self).__init__()
        self.root = root
        self.name = name
        self.num_of_clients = num_of_clients
        self.tr_frac = tr_frac
        self.feature_partition = feature_partition
        self.seed = seed
        self.data_dict = {}
        self.data = {}

        if download:
            self.download()

        self._get_data()

    base_folder = 'adult'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
    raw_file = ['adult.data', 'adult.test']

    def _get_data(self):
        fpath = os.path.join(self.root, self.base_folder)
        train_file = osp.join(fpath, 'adult.data')
        test_file = osp.join(fpath, 'adult.test')
        train_data = self._read_raw(train_file)
        test_data = self._read_raw(test_file)
        train_data, test_data = self.process(train_data, test_data)
        self._partition_data(train_data, test_data)

    def _read_raw(self, file_path):
        data = pd.read_csv(file_path, header=None)
        return data

    def process(self, train_set, test_set):
        """https://cloud.tencent.com/developer/article/1338337"""
        col_labels = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
            'wage_class'
        ]
        # test_set = test_set.drop

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
        return train_set, test_set

    def _partition_data(self, train_set, test_set):
        train_set = train_set.values
        test_set = test_set.values
        x, y = train_set[:, :-1], train_set[:, -1]
        test_x, test_y = test_set[:, :-1], test_set[:, -1]
        test_data = {'x': test_x, 'y': test_y}

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

    def _check_existence(self, file):
        fpath = os.path.join(self.root, self.base_folder, file)
        return osp.exists(fpath)

    def download(self):
        for file in self.raw_file:
            if self._check_existence(file):
                logger.info(file + " already downloaded and verified")
            # print("Files already downloaded and verified")
            else:
                download_and_extract_archive(f'{self.url}/{file}',
                                             os.path.join(
                                                 self.root, self.base_folder),
                                             filename=file)
