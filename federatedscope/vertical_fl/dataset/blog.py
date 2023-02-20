import glob
import logging
import os
import os.path as osp

import numpy as np
import pandas as pd

from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class Blog(object):
    """
    BlogFeedback Data Set
    (https://archive.ics.uci.edu/ml/datasets/BlogFeedback)

    Data Set Information:
    This data originates from blog posts. The raw HTML-documents
    of the blog posts were crawled and processed.
    The prediction task associated with the data is the prediction
    of the number of comments in the upcoming 24 hours. In order
    to simulate this situation, we choose a basetime (in the past)
    and select the blog posts that were published at most
    72 hours before the selected base date/time. Then, we calculate
    all the features of the selected blog posts from the information
    that was available at the basetime, therefore each instance
    corresponds to a blog post. The target is the number of
    comments that the blog post received in the next 24 hours
    relative to the basetime.

    Number of Instances: 60021
    Number of Attributes: 281, the last one is the number of comments
                          in the next 24 hours
    Training set: 'blogData_train.csv', 52397 instances
    Testing set: 'blogData_test*.csv', 60 files, 7624 instances totally

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
    base_folder = 'blogfeedback'
    url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com/BlogFeedback.zip'
    raw_file = 'BlogFeedback.zip'

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
        super(Blog, self).__init__()
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
        if not self._check_existence():
            raise RuntimeError("Dataset not found or corrupted." +
                               "You can use download=True to download it")

        self._get_data()
        self._partition_data()

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

        if self.data_size_for_debug != 0:
            subset_size = min(len(train_data), self.data_size_for_debug)
            np.random.shuffle(train_data)
            train_data = train_data[:subset_size]

        self.data_dict['train'] = train_data
        self.data_dict['test'] = test_data

    def _read_raw(self, file_path):
        data = pd.read_csv(file_path, header=None, usecols=list(range(281)))
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
        x = self.data_dict['train'][:, :self.feature_partition[-1]]
        y = self.data_dict['train'][:, self.feature_partition[-1]]
        test_data = dict()
        test_data['x'] = self.data_dict['test'][:, :self.feature_partition[-1]]
        test_data['y'] = self.data_dict['test'][:, self.feature_partition[-1]]

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
