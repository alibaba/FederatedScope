import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import pickle as pkl
import numpy as np
from federatedscope.register import register_data
from federatedscope.core.auxiliaries.splitter_builder import get_splitter


class SimCLRTransform():
    r"""
    Data Augmentations of SimCLR refer from
    https://github.com/akhilmathurs/orchestra/blob/main/utils.py
    Arguments:
        is_sup (bool): the transform for supervised learning
        or contrastive learning.
    :returns:
        torch.tensor: one output for supervised learning.
    :returns:
        torch.tensor: two output for contrastive learning
        torch.tensor: two output for contrastive learning
    """
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size,
                                scale=(0.5, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                          p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        if (self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2


def Cifar4CL(config):
    r"""
    generate Cifar10 Dataset transform and split dict for contrastive learning
    return {
                'client_id': {
                    'train': DataLoader(),
                    'test': DataLoader(),
                    'val': DataLoader()
                }
            }
    """
    transform_train = SimCLRTransform(is_sup=False, image_size=32)

    path = config.data.root

    data_train = CIFAR10(path,
                         train=True,
                         download=True,
                         transform=transform_train)
    data_test = CIFAR10(path,
                        train=False,
                        download=True,
                        transform=transform_train)

    # Split data into dict
    data_dict = dict()
    data_val = data_train

    data_dict = {'train': data_train, 'val': data_val, 'test': data_test}
    data_split_tuple = (data_dict.get('train'), data_dict.get('val'),
                        data_dict.get('test'))

    config = config
    return data_split_tuple, config


def Cifar4LP(config):
    r"""
    generate Cifar10 Dataset transform and split dict for linear prob
    evaluation of contrastive learning
    return {
                'client_id': {
                    'train': DataLoader(),
                    'test': DataLoader(),
                    'val': DataLoader()
                }
            }
    """
    transform_train = T.Compose([
        T.RandomResizedCrop(32,
                            scale=(0.5, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    path = config.data.root

    data_train = CIFAR10(path,
                         train=True,
                         download=True,
                         transform=transform_train)
    data_val = CIFAR10(path,
                       train=True,
                       download=True,
                       transform=transform_test)
    data_test = CIFAR10(path,
                        train=False,
                        download=True,
                        transform=transform_test)

    # Split data into dict
    data_dict = dict()
    data_val = data_train

    data_dict = {'train': data_train, 'val': data_val, 'test': data_test}
    data_split_tuple = (data_dict.get('train'), data_dict.get('val'),
                        data_dict.get('test'))

    config = config
    return data_split_tuple, config


def load_cifar_dataset(config):
    if config.data.type == "Cifar4CL":
        data, modified_config = Cifar4CL(config)
        return data, modified_config
    elif config.data.type == "Cifar4LP":
        data, modified_config = Cifar4LP(config)
        return data, modified_config
