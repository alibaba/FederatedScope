from torch.utils.data import DataLoader

from federatedscope.cv.dataset.leaf_cv import LEAF_CV
from federatedscope.core.auxiliaries.transform_builder import get_transform


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
    batch_size = config.data.batch_size
    transforms_funcs = get_transform(config, 'torchvision')

    if name in ['femnist', 'celeba']:
        dataset = LEAF_CV(root=path,
                          name=name,
                          s_frac=config.data.subsample,
                          tr_frac=splits[0],
                          val_frac=splits[1],
                          seed=1234,
                          **transforms_funcs)
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
