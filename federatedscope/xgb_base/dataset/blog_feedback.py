import glob
import logging
import os
import os.path as osp

import numpy as np
import pandas as pd

from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class BlogFeedback:
    """
    label 0 - 279 are features and 280 is the value
    """
    def __init__(self,
                 root,
                 name,
                 num_of_clients,
                 feature_partition,
                 tr_frac=0.9,
                 download=True,
                 seed=123):
        super(BlogFeedback, self).__init__()
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
        if not self._check_existence():
            raise RuntimeError("Dataset not found or corrupted." +
                               "You can use download=True to download it")

        self._get_data()
        self._partition_data()

    base_folder = 'blogfeedback'
    url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com/BlogFeedback.zip'
    raw_file = 'BlogFeedback.zip'

    def _get_data(self):
        fpath = os.path.join(self.root, self.base_folder)
        train_file = osp.join(fpath, 'blogData_train.csv')
        train_data = self._read_raw(train_file)
        test_files = glob.glob(osp.join(fpath, "blogData_test*.csv"))
        test_files.sort()

        flag = 0
        for f in test_files:
            f_data = self._read_raw(f)
            if flag == 0:
                test_data = f_data
                flag = 1
            else:
                test_data = np.concatenate((test_data, f_data), axis=0)

        self.data_dict['train'] = train_data
        self.data_dict['test'] = test_data

    def _read_raw(self, file_path):
        data = pd.read_csv(file_path, header=None, usecols=list(range(281)))
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
        x = self.data_dict['train'][:, :self.feature_partition[-1]]
        y = self.data_dict['train'][:, self.feature_partition[-1]]
        test_data = dict()
        test_data['x'] = self.data_dict['test'][:, :self.feature_partition[-1]]
        test_data['y'] = self.data_dict['test'][:, self.feature_partition[-1]]

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
