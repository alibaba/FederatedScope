import os
import random
import json

import torch
import math

import os.path as osp

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from federatedscope.core.data.utils import save_local_data, download_url
from federatedscope.cv.dataset.leaf import LEAF, LocalDataset
from federatedscope.nlp.dataset.utils import *


class LEAF_TWITTER(LEAF):
    """
    LEAF NLP dataset from

    leaf.cmu.edu

    Arguments:
        root (str): root path.
        name (str): name of dataset, ‘shakespeare’ or ‘xxx’.
        s_frac (float): fraction of the dataset to be used; default=0.3.
        tr_frac (float): train set proportion for each task; default=0.8.
        val_frac (float): valid set proportion for each task; default=0.0.
        transform: transform for x.
        target_transform: transform for y.

    """
    def __init__(self,
                 root,
                 name='twitter',
                 max_len=140,
                 s_frac=0.3,
                 tr_frac=0.8,
                 val_frac=0.0,
                 seed=123,
                 transform=None,
                 target_transform=None):
        self.root = root
        self.name = name
        self.s_frac = s_frac
        self.tr_frac = tr_frac
        self.val_frac = val_frac
        self.seed = seed
        self.max_len = max_len
        if name != 'twitter':
            raise ValueError('`name` should be `twitter`.')
        else:
            if not os.path.exists(
                    osp.join(osp.join(root, name, 'raw'), 'embs.json')):
                self.download()
                self.extract()
            print('Loading embs...')
            with open(osp.join(osp.join(root, name, 'raw'), 'embs.json'),
                      'r') as inf:
                embs = json.load(inf)
            self.id2word = embs['vocab']
            self.word2id = {v: k for k, v in enumerate(self.id2word)}
        super(LEAF_TWITTER, self).__init__(root, name, transform,
                                           target_transform)
        files = os.listdir(self.processed_dir)
        files = [f for f in files if f.startswith('task_')]
        if len(files):
            # Sort by idx
            files.sort(key=lambda k: int(k[5:]))

            for file in files:
                train_data, train_targets = torch.load(
                    osp.join(self.processed_dir, file, 'train.pt'))
                self.data_dict[int(file[5:])] = {
                    'train': (train_data, train_targets)
                }
                if osp.exists(osp.join(self.processed_dir, file, 'test.pt')):
                    test_data, test_targets = torch.load(
                        osp.join(self.processed_dir, file, 'test.pt'))
                    self.data_dict[int(file[5:])]['test'] = (test_data,
                                                             test_targets)
                if osp.exists(osp.join(self.processed_dir, file, 'val.pt')):
                    val_data, val_targets = torch.load(
                        osp.join(self.processed_dir, file, 'val.pt'))
                    self.data_dict[int(file[5:])]['val'] = (val_data,
                                                            val_targets)
        else:
            raise RuntimeError(
                'Please delete ‘processed’ folder and try again!')

    @property
    def raw_file_names(self):
        names = [f'{self.name}_all_data.zip']
        return names

    def download(self):
        # Download to `self.raw_dir`.
        url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            download_url(f'{url}/{name}', self.raw_dir)

    def _to_bag_of_word(self, text):
        bag = np.zeros(len(self.word2id))
        for i in text:
            if i != -1:
                bag[i] += 1
            else:
                break
        text = torch.FloatTensor(bag)

        return text

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        :returns:
            dict: {'train':Dataset,
                   'test':Dataset,
                   'val':Dataset}
            where target is the target class.
        """
        text_dict = {}
        data = self.data_dict[index]
        for key in data:
            text_dict[key] = []
            texts, targets = data[key]
            if self.transform:
                text_dict[key] = LocalDataset(texts, targets, None,
                                              self.transform,
                                              self.target_transform)
            else:
                text_dict[key] = LocalDataset(texts, targets, None,
                                              self._to_bag_of_word,
                                              self.target_transform)

        return text_dict

    def tokenizer(self, data, targets):
        # [ID, Date, Query, User, Content]
        processed_data = []
        for raw_text in data:
            ids = [
                self.word2id[w] if w in self.word2id else 0
                for w in split_line(raw_text[4])
            ]
            if len(ids) < self.max_len:
                ids += [-1] * (self.max_len - len(ids))
            else:
                ids = ids[:self.max_len]
            processed_data.append(ids)
        targets = [target_to_binary(raw_target) for raw_target in targets]

        return processed_data, targets

    def process(self):
        raw_path = osp.join(self.raw_dir, "all_data")
        files = os.listdir(raw_path)
        files = [f for f in files if f.endswith('.json')]

        print("Preprocess data (Please leave enough space)...")

        idx = 0
        for num, file in enumerate(files):
            with open(osp.join(raw_path, file), 'r') as f:
                raw_data = json.load(f)
            user_list = list(raw_data['user_data'].keys())
            n_tasks = math.ceil(len(user_list) * self.s_frac)
            random.shuffle(user_list)
            user_list = user_list[:n_tasks]
            for user in tqdm(user_list):
                data, targets = raw_data['user_data'][user]['x'], raw_data[
                    'user_data'][user]['y']

                # Tokenize
                data, targets = self.tokenizer(data, targets)

                if len(data) > 2:
                    data = torch.LongTensor(np.stack(data))
                    targets = torch.LongTensor(np.stack(targets))
                else:
                    data = torch.LongTensor(data)
                    targets = torch.LongTensor(targets)

                try:
                    train_data, test_data, train_targets, test_targets = \
                        train_test_split(
                            data,
                            targets,
                            train_size=self.tr_frac,
                            random_state=self.seed
                        )
                except ValueError:
                    train_data = data
                    train_targets = targets
                    test_data, test_targets = None, None

                if self.val_frac > 0:
                    try:
                        val_data, test_data, val_targets, test_targets = \
                            train_test_split(
                                test_data,
                                test_targets,
                                train_size=self.val_frac / (1. - self.tr_frac),
                                random_state=self.seed
                            )
                    except:
                        val_data, val_targets = None, None

                else:
                    val_data, val_targets = None, None
                save_path = osp.join(self.processed_dir, f"task_{idx}")
                os.makedirs(save_path, exist_ok=True)

                save_local_data(dir_path=save_path,
                                train_data=train_data,
                                train_targets=train_targets,
                                test_data=test_data,
                                test_targets=test_targets,
                                val_data=val_data,
                                val_targets=val_targets)
                idx += 1
