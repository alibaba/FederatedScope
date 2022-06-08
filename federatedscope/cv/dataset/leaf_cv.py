import os
import random
import json
import torch
import math

import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from federatedscope.core.auxiliaries.utils import save_local_data, download_url
from federatedscope.cv.dataset.leaf import LEAF

IMAGE_SIZE = {'femnist': (28, 28), 'celeba': (84, 84, 3)}
MODE = {'femnist': 'L', 'celeba': 'RGB'}


class LEAF_CV(LEAF):
    """
    LEAF CV dataset from "LEAF: A Benchmark for Federated Settings"

    leaf.cmu.edu

    Arguments:
        root (str): root path.
        name (str): name of dataset, ‘femnist’ or ‘celeba’.
        s_frac (float): fraction of the dataset to be used; default=0.3.
        tr_frac (float): train set proportion for each task; default=0.8.
        val_frac (float): valid set proportion for each task; default=0.0.
        train_tasks_frac (float): fraction of test tasks; default=1.0.
        transform: transform for x.
        target_transform: transform for y.

    """
    def __init__(self,
                 root,
                 name,
                 s_frac=0.3,
                 tr_frac=0.8,
                 val_frac=0.0,
                 train_tasks_frac=1.0,
                 seed=123,
                 transform=None,
                 target_transform=None):
        self.s_frac = s_frac
        self.tr_frac = tr_frac
        self.val_frac = val_frac
        self.seed = seed
        self.train_tasks_frac = train_tasks_frac
        super(LEAF_CV, self).__init__(root, name, transform, target_transform)
        files = os.listdir(self.processed_dir)
        files = [f for f in files if f.startswith('task_')]
        if len(files):
            # Sort by idx
            files.sort(key=lambda k: int(k[5:]))

            for file in files:
                train_data, train_targets = torch.load(
                    osp.join(self.processed_dir, file, 'train.pt'))
                test_data, test_targets = torch.load(
                    osp.join(self.processed_dir, file, 'test.pt'))
                self.data_dict[int(file[5:])] = {
                    'train': (train_data, train_targets),
                    'test': (test_data, test_targets)
                }
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

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        :returns:
            dict: {'train':[(image, target)],
                   'test':[(image, target)],
                   'val':[(image, target)]}
            where target is the target class.
        """
        img_dict = {}
        data = self.data_dict[index]
        for key in data:
            img_dict[key] = []
            imgs, targets = data[key]
            for idx in range(targets.shape[0]):
                img = np.resize(imgs[idx].numpy().astype(np.uint8),
                                IMAGE_SIZE[self.name])
                img = Image.fromarray(img, mode=MODE[self.name])
                target = targets[idx]
                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                img_dict[key].append((img, targets[idx]))

        return img_dict

    def process(self):
        raw_path = osp.join(self.raw_dir, "all_data")
        files = os.listdir(raw_path)
        files = [f for f in files if f.endswith('.json')]

        n_tasks = math.ceil(len(files) * self.s_frac)
        random.shuffle(files)
        files = files[:n_tasks]

        print("Preprocess data (Please leave enough space)...")

        idx = 0
        for num, file in enumerate(tqdm(files)):

            with open(osp.join(raw_path, file), 'r') as f:
                raw_data = json.load(f)

            # Numpy to Tensor
            for writer, v in raw_data['user_data'].items():
                data, targets = v['x'], v['y']

                if len(v['x']) > 2:
                    data = torch.tensor(np.stack(data))
                    targets = torch.LongTensor(np.stack(targets))
                else:
                    data = torch.tensor(data)
                    targets = torch.LongTensor(targets)

                train_data, test_data, train_targets, test_targets =\
                    train_test_split(
                        data,
                        targets,
                        train_size=self.tr_frac,
                        random_state=self.seed
                    )

                if self.val_frac > 0:
                    val_data, test_data, val_targets, test_targets = \
                        train_test_split(
                            test_data,
                            test_targets,
                            train_size=self.val_frac / (1.-self.tr_frac),
                            random_state=self.seed
                        )

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
