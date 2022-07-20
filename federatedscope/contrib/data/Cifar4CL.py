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



class SimCLRTransform():
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 

def MyData(config):
    
    transform_train = SimCLRTransform(is_sup=False, image_size=32)
    transform_test = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    
    data_train = CIFAR10(f"{data_dir}/dataset/CIFAR10", train=True, download=False, transform=transform_train)
    data_val = CIFAR10(f"{data_dir}/dataset/CIFAR10", train=True, download=False, transform=transform_test)
    data_test = CIFAR10(f"{data_dir}/dataset/CIFAR10", train=False, download=False, transform=transform_test)
    
          # Split data into dict
    data_dict = dict()
    train_per_client = len(data_train) // config.federate.client_num
    test_per_client = len(data_test) // config.federate.client_num
    
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {
              'train':
              DataLoader([
                  data_train[i]
                  for i in range((client_idx - 1) *
                                 train_per_client, client_idx * train_per_client)
              ],
                         config.data.batch_size,
                         shuffle=config.data.shuffle),
              'test':
              DataLoader([
                  data_test[i]
                  for i in range((client_idx - 1) * test_per_client, client_idx *
                                 test_per_client)
              ],
                         config.data.batch_size,
                         shuffle=False)
          }
        data_dict[client_idx] = dataloader_dict
    r"""

    Returns:
            data:
                {
                    '{client_id}': {
                        'train': Dataset or DataLoader,
                        'test': Dataset or DataLoader,
                        'val': Dataset or DataLoader
                    }
                }
            config:
                cfg_node
    """
    data = data_dict
    config = config
    return data, config

from federatedscope.register import register_data

def call_my_data(config):
    if config.data.type == "Cifar10":
        data, modified_config = MyData(config)
        return data, modified_config


register_data("Cifar10", call_my_data)
