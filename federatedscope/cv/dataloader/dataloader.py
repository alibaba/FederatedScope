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

    # Convert list to dict
    data_dict = dict()
    for client_idx in range(1, client_num + 1):
        data_dict[client_idx] = dataset[client_idx - 1]

    return data_dict, config
