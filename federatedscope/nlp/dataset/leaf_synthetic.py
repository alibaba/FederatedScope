import os
import pickle
import argparse
import torch
import numpy as np
import os.path as osp

from sklearn.utils import shuffle
from torch.utils.data import Dataset

from federatedscope.core.data.utils import save_local_data
from federatedscope.cv.dataset.leaf import LEAF


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


class LEAF_SYNTHETIC(LEAF):
    """SYNTHETIC dataset from "Federated Multi-Task Learning
    under a Mixture of Distributions"

    Source: https://github.com/omarfoq/FedEM/tree/main/data/synthetic

    Arguments:
        root (str): root path.
        name (str): name of dataset, `SYNTHETIC`.
        n_components (int): number of mixture components, default=3.
        n_task (int): number of tasks/clients, default = 300.
        n_test (int): size of test set, default=5,000.
        n_val (int): size of validation set, default=5,000.
        dim (int): dimension of the data, default=150.
        noise_level (float): proportion of noise, default=0.1.
        alpha (float): alpha of LDA, default=0.4.
        box (list): box of `x`, default=(-1.0, 1.0).

    """
    def __init__(self,
                 root,
                 name='synthetic',
                 n_components=3,
                 n_tasks=300,
                 n_test=5000,
                 n_val=5000,
                 dim=150,
                 noise_level=0.1,
                 alpha=0.4,
                 box=(-1.0, 1.0),
                 uniform_marginal=True):

        self.root = root
        self.n_components = n_components
        self.n_tasks = n_tasks
        self.n_test = n_test
        self.n_val = n_val
        self.dim = dim
        self.noise_level = noise_level
        self.alpha = alpha * np.ones(n_components)
        self.box = box
        self.uniform_marginal = uniform_marginal
        self.num_samples = self.get_num_samples(self.n_tasks)

        self.theta = np.zeros((self.n_components, self.dim))
        self.mixture_weights = np.zeros((self.n_tasks, self.n_components))

        self.generate_mixture_weights()
        self.generate_components()

        super(LEAF_SYNTHETIC, self).__init__(root, name, None, None)
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

    def download(self):
        pass

    def extract(self):
        pass

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        :returns:
            dict: {'train':[(x, target)],
                   'test':[(x, target)],
                   'val':[(x, target)]}
            where target is the target class.
        """
        text_dict = {}
        data = self.data_dict[index]
        for key in data:
            text_dict[key] = []
            texts, targets = data[key]
            for idx in range(targets.shape[0]):
                text = texts[idx]
                text_dict[key].append((text, targets[idx]))

        return text_dict

    def generate_mixture_weights(self):
        for task_id in range(self.n_tasks):
            self.mixture_weights[task_id] = np.random.dirichlet(
                alpha=self.alpha)

    def generate_components(self):
        self.theta = np.random.uniform(self.box[0],
                                       self.box[1],
                                       size=(self.n_components, self.dim))

    def generate_data(self, task_id, n_samples=10000):
        latent_variable_count = np.random.multinomial(
            n_samples, self.mixture_weights[task_id])
        y = np.zeros(n_samples)

        if self.uniform_marginal:
            x = np.random.uniform(self.box[0],
                                  self.box[1],
                                  size=(n_samples, self.dim))
        else:
            raise NotImplementedError(
                "Only uniform marginal is available for the moment")

        current_index = 0
        for component_id in range(self.n_components):
            y_hat = x[current_index:current_index +
                      latent_variable_count[component_id]] @ self.theta[
                          component_id]
            noise = np.random.normal(size=latent_variable_count[component_id],
                                     scale=self.noise_level)
            y[current_index:current_index +
              latent_variable_count[component_id]] = np.round(
                  sigmoid(y_hat + noise)).astype(int)

        return shuffle(x.astype(np.float32), y.astype(np.int64))

    def save_metadata(self, path_):
        metadata = dict()
        metadata["mixture_weights"] = self.mixture_weights
        metadata["theta"] = self.theta

        with open(path_, 'wb') as f:
            pickle.dump(metadata, f)

    def get_num_samples(self,
                        num_tasks,
                        min_num_samples=50,
                        max_num_samples=1000):
        num_samples = np.random.lognormal(4, 2, num_tasks).astype(int)
        num_samples = [
            min(s + min_num_samples, max_num_samples) for s in num_samples
        ]
        return num_samples

    def process(self):
        for task_id in range(self.n_tasks):
            save_path = os.path.join(self.processed_dir, f"task_{task_id}")
            os.makedirs(save_path, exist_ok=True)

            train_data, train_targets = self.generate_data(
                task_id, self.num_samples[task_id])
            test_data, test_targets = self.generate_data(task_id, self.n_test)

            if self.n_val > 0:
                val_data, val_targets = self.generate_data(task_id, self.n_val)
            else:
                val_data, val_targets = None, None
            save_local_data(dir_path=save_path,
                            train_data=train_data,
                            train_targets=train_targets,
                            test_data=test_data,
                            test_targets=test_targets,
                            val_data=val_data,
                            val_targets=val_targets)
