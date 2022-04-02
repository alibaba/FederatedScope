import torch
import numpy as np

from collections import Iterable
from torchvision import transforms
from torch.utils.data import DataLoader

from federatedscope.cv.dataset.leaf_cv import LEAF_CV


def get_transforms(config):

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    if config.data.transforms is None:
        return None
    else:
        if isinstance(config.data.transforms, Iterable):
            trans = list()
            for type in config.data.transforms:
                if isinstance(type, str):
                    if hasattr(transforms, type):
                        trans.append(getattr(transforms, type)())
                    else:
                        raise NotImplementedError(
                            'Transform {} not implement'.format(type))
                elif isinstance(type, list):
                    pass
            return transforms.Compose(trans)
        else:
            raise TypeError()


def load_cv_dataset(config=None):
    r"""
    return {
                'client_id': {
                    'train': DataLoader(),
                    'test': DataLoader(),
                    'val': DataLoader()
                }
            }
    """
    splits = config.data.splits

    path = config.data.root
    name = config.data.type.lower()
    client_num = config.federate.client_num
    batch_size = config.data.batch_size

    transform = get_transforms(config)

    if name in ['femnist', 'celeba']:
        dataset = LEAF_CV(root=path,
                          name=name,
                          s_frac=config.data.subsample,
                          tr_frac=splits[0],
                          val_frac=splits[1],
                          seed=1234,
                          transform=transform)
    else:
        raise ValueError(f'No dataset named: {name}!')

    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_local_dict = dict()
    for client_idx in range(client_num):
        dataloader = {
            'train': DataLoader(dataset[client_idx]['train'],
                                batch_size,
                                shuffle=config.data.shuffle,
                                num_workers=config.data.num_workers),
            'test': DataLoader(dataset[client_idx]['test'],
                               batch_size,
                               shuffle=False,
                               num_workers=config.data.num_workers)
        }
        if 'val' in dataset[client_idx]:
            dataloader['val'] = DataLoader(dataset[client_idx]['val'],
                                           batch_size,
                                           shuffle=False,
                                           num_workers=config.data.num_workers)

        data_local_dict[client_idx + 1] = dataloader

    return data_local_dict, config
