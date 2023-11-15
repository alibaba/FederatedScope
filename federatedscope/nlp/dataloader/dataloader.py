from federatedscope.nlp.dataset.leaf_nlp import LEAF_NLP
from federatedscope.nlp.dataset.leaf_twitter import LEAF_TWITTER
from federatedscope.nlp.dataset.leaf_synthetic import LEAF_SYNTHETIC
from federatedscope.core.auxiliaries.transform_builder import get_transform


def load_nlp_dataset(config=None):
    """
    Return the dataset of ``shakespeare``, ``subreddit``, ``twitter``, \
    or ``synthetic``.

    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        FL dataset dict, with ``client_id`` as key.

    Note:
      ``load_nlp_dataset()`` will return a dict as shown below:
        ```
        {'client_id': {'train': dataset, 'test': dataset, 'val': dataset}}
        ```
    """
    splits = config.data.splits

    path = config.data.root
    name = config.data.type.lower()
    transforms_funcs, _, _ = get_transform(config, 'torchtext')

    if name in ['shakespeare', 'subreddit']:
        dataset = LEAF_NLP(root=path,
                           name=name,
                           s_frac=config.data.subsample,
                           tr_frac=splits[0],
                           val_frac=splits[1],
                           seed=config.seed,
                           **transforms_funcs)
    elif name == 'twitter':
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
    data_dict = dict()
    for client_idx in range(1, client_num + 1):
        data_dict[client_idx] = dataset[client_idx - 1]

    return data_dict, config
