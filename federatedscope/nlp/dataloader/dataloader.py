from torch.utils.data import DataLoader

from federatedscope.nlp.dataset.leaf_nlp import LEAF_NLP
from federatedscope.nlp.dataset.leaf_twitter import LEAF_TWITTER
from federatedscope.nlp.dataset.leaf_synthetic import LEAF_SYNTHETIC
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.core.interface.base_data import ClientData, \
    StandaloneDataDict


def load_nlp_dataset(config=None, client_cfgs=None):
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
    transforms_funcs = get_transform(config, 'torchtext')

    if name in ['shakespeare', 'subreddit']:
        dataset = LEAF_NLP(root=path,
                           name=name,
                           s_frac=config.data.subsample,
                           tr_frac=splits[0],
                           val_frac=splits[1],
                           seed=config.seed,
                           **transforms_funcs)
    if name == 'twitter':
        dataset = LEAF_TWITTER(root=path,
                               name='twitter',
                               s_frac=config.data.subsample,
                               tr_frac=splits[0],
                               val_frac=splits[1],
                               seed=config.seed,
                               **transforms_funcs)
    elif name == 'synthetic':
        dataset = LEAF_SYNTHETIC(root=path)
    else:
        raise ValueError(f'No dataset named: {name}!')

    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_local_dict = dict()
    for client_idx in range(1, client_num + 1):
        if client_cfgs is not None:
            client_cfg = config.clone()
            client_cfg.merge_from_other_cfg(
                client_cfgs.get(f'client_{client_idx}'))
        else:
            client_cfg = config
        client_data = ClientData(DataLoader,
                                 client_cfg,
                                 train=dataset[client_idx - 1].get('train'),
                                 val=dataset[client_idx - 1].get('val'),
                                 test=dataset[client_idx - 1].get('test'))
        data_local_dict[client_idx] = client_data

    return StandaloneDataDict(data_local_dict, config), config
