import pickle
import logging
import numpy as np
from collections import defaultdict

import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.data import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.data`, some modules are not '
        f'available.')


def load_toy_data(config=None):

    generate = config.federate.mode.lower() == 'standalone'

    def _generate_data(client_num=5,
                       instance_num=1000,
                       feature_num=5,
                       save_data=False):
        """
        Generate data in FedRunner format
        Args:
            client_num:
            instance_num:
            feature_num:
            save_data:

        Returns:
            {
                '{client_id}': {
                    'train': {
                        'x': ...,
                        'y': ...
                    },
                    'test': {
                        'x': ...,
                        'y': ...
                    },
                    'val': {
                        'x': ...,
                        'y': ...
                    }
                }
            }

        """
        weights = np.random.normal(loc=0.0, scale=1.0, size=feature_num)
        bias = np.random.normal(loc=0.0, scale=1.0)
        data = dict()
        for each_client in range(1, client_num + 1):
            data[each_client] = dict()
            client_x = np.random.normal(loc=0.0,
                                        scale=0.5 * each_client,
                                        size=(instance_num, feature_num))
            client_y = np.sum(client_x * weights, axis=-1) + bias
            client_y = np.expand_dims(client_y, -1)
            client_data = {'x': client_x, 'y': client_y}
            data[each_client]['train'] = client_data

        # test data
        test_x = np.random.normal(loc=0.0,
                                  scale=1.0,
                                  size=(instance_num, feature_num))
        test_y = np.sum(test_x * weights, axis=-1) + bias
        test_y = np.expand_dims(test_y, -1)
        test_data = {'x': test_x, 'y': test_y}
        for each_client in range(1, client_num + 1):
            data[each_client]['test'] = test_data

        # val data
        val_x = np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(instance_num, feature_num))
        val_y = np.sum(val_x * weights, axis=-1) + bias
        val_y = np.expand_dims(val_y, -1)
        val_data = {'x': val_x, 'y': val_y}
        for each_client in range(1, client_num + 1):
            data[each_client]['val'] = val_data

        # server_data
        data[0] = dict()
        data[0]['train'] = None
        data[0]['val'] = val_data
        data[0]['test'] = test_data

        if save_data:
            # server_data = dict()
            save_client_data = dict()

            for client_idx in range(0, client_num + 1):
                if client_idx == 0:
                    filename = 'data/server_data'
                else:
                    filename = 'data/client_{:d}_data'.format(client_idx)
                with open(filename, 'wb') as f:
                    save_client_data['train'] = {
                        k: v.tolist()
                        for k, v in data[client_idx]['train'].items()
                    }
                    save_client_data['val'] = {
                        k: v.tolist()
                        for k, v in data[client_idx]['val'].items()
                    }
                    save_client_data['test'] = {
                        k: v.tolist()
                        for k, v in data[client_idx]['test'].items()
                    }
                    pickle.dump(save_client_data, f)

        return data

    if generate:
        data = _generate_data(client_num=config.federate.client_num,
                              save_data=config.eval.save_data)
    else:
        with open(config.distribute.data_file, 'rb') as f:
            data = pickle.load(f)
        for key in data.keys():
            data[key] = {k: np.asarray(v)
                         for k, v in data[key].items()
                         } if data[key] is not None else None

    return data, config


def load_external_data(config=None):
    r""" Based on the configuration file, this function imports external
    datasets and applies train/valid/test splits and split by some specific
    `splitter` into the standard FederatedScope input data format.

    Args:
        config: `CN` from `federatedscope/core/configs/config.py`

    Returns:
        data_local_dict: dict of split dataloader.
                        Format:
                            {
                                'client_id': {
                                    'train': DataLoader(),
                                    'test': DataLoader(),
                                    'val': DataLoader()
                                }
                            }
        modified_config: `CN` from `federatedscope/core/configs/config.py`,
        which might be modified in the function.

    """

    import torch
    import inspect
    from importlib import import_module
    from torch.utils.data import DataLoader
    from federatedscope.core.auxiliaries.splitter_builder import get_splitter
    from federatedscope.core.auxiliaries.transform_builder import get_transform

    def get_func_args(func):
        sign = inspect.signature(func).parameters.values()
        sign = set([val.name for val in sign])
        return sign

    def filter_dict(func, kwarg):
        sign = get_func_args(func)
        common_args = sign.intersection(kwarg.keys())
        filtered_dict = {key: kwarg[key] for key in common_args}
        return filtered_dict

    def load_torchvision_data(name, splits=None, config=None):
        dataset_func = getattr(import_module('torchvision.datasets'), name)
        transform_funcs = get_transform(config, 'torchvision')
        if config.data.args:
            raw_args = config.data.args[0]
        else:
            raw_args = {}
        if 'download' not in raw_args.keys():
            raw_args.update({'download': True})
        filtered_args = filter_dict(dataset_func.__init__, raw_args)
        func_args = get_func_args(dataset_func.__init__)

        # Perform split on different dataset
        if 'train' in func_args:
            # Split train to (train, val)
            dataset_train = dataset_func(root=config.data.root,
                                         train=True,
                                         **filtered_args,
                                         **transform_funcs)
            dataset_val = None
            dataset_test = dataset_func(root=config.data.root,
                                        train=False,
                                        **filtered_args,
                                        **transform_funcs)
            if splits:
                train_size = int(splits[0] * len(dataset_train))
                val_size = len(dataset_train) - train_size
                lengths = [train_size, val_size]
                dataset_train, dataset_val = \
                    torch.utils.data.dataset.random_split(dataset_train,
                                                          lengths)

        elif 'split' in func_args:
            # Use raw split
            dataset_train = dataset_func(root=config.data.root,
                                         split='train',
                                         **filtered_args,
                                         **transform_funcs)
            dataset_val = dataset_func(root=config.data.root,
                                       split='valid',
                                       **filtered_args,
                                       **transform_funcs)
            dataset_test = dataset_func(root=config.data.root,
                                        split='test',
                                        **filtered_args,
                                        **transform_funcs)
        elif 'classes' in func_args:
            # Use raw split
            dataset_train = dataset_func(root=config.data.root,
                                         classes='train',
                                         **filtered_args,
                                         **transform_funcs)
            dataset_val = dataset_func(root=config.data.root,
                                       classes='valid',
                                       **filtered_args,
                                       **transform_funcs)
            dataset_test = dataset_func(root=config.data.root,
                                        classes='test',
                                        **filtered_args,
                                        **transform_funcs)
        else:
            # Use config.data.splits
            dataset = dataset_func(root=config.data.root,
                                   **filtered_args,
                                   **transform_funcs)
            train_size = int(splits[0] * len(dataset))
            val_size = int(splits[1] * len(dataset))
            test_size = len(dataset) - train_size - val_size
            lengths = [train_size, val_size, test_size]
            dataset_train, dataset_val, dataset_test = \
                torch.utils.data.dataset.random_split(dataset, lengths)

        data_dict = {
            'train': dataset_train,
            'val': dataset_val,
            'test': dataset_test
        }

        return data_dict

    def load_torchtext_data(name, splits=None, config=None):
        from torch.nn.utils.rnn import pad_sequence
        from federatedscope.nlp.dataset.utils import label_to_index

        dataset_func = getattr(import_module('torchtext.datasets'), name)
        if config.data.args:
            raw_args = config.data.args[0]
        else:
            raw_args = {}
        assert 'max_len' in raw_args, "Miss key 'max_len' in " \
                                      "`config.data.args`."
        filtered_args = filter_dict(dataset_func.__init__, raw_args)
        dataset = dataset_func(root=config.data.root, **filtered_args)

        # torchtext.transforms requires >= 0.12.0 and torch = 1.11.0,
        # so we do not use `get_transform` in torchtext.

        # Merge all data and tokenize
        x_list = []
        y_list = []
        for data_iter in dataset:
            data, targets = [], []
            for i, item in enumerate(data_iter):
                data.append(item[1])
                targets.append(item[0])
            x_list.append(data)
            y_list.append(targets)

        x_all, y_all = [], []
        for i in range(len(x_list)):
            x_all += x_list[i]
            y_all += y_list[i]

        if config.model.type.endswith('transformers'):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.type.split('@')[0])

            x_all = tokenizer(x_all,
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=raw_args['max_len'])
            data = [{key: value[i]
                     for key, value in x_all.items()}
                    for i in range(len(next(iter(x_all.values()))))]
            if 'classification' in config.model.task.lower():
                targets = label_to_index(y_all)
            else:
                y_all = tokenizer(y_all,
                                  return_tensors='pt',
                                  padding=True,
                                  truncation=True,
                                  max_length=raw_args['max_len'])
                targets = [{key: value[i]
                            for key, value in y_all.items()}
                           for i in range(len(next(iter(y_all.values()))))]
        else:
            from torchtext.data import get_tokenizer
            tokenizer = get_tokenizer("basic_english")
            if len(config.data.transform) == 0:
                raise ValueError(
                    "`transform` must be one pretrained Word Embeddings from \
                    ['GloVe', 'FastText', 'CharNGram']")
            if len(config.data.transform) == 1:
                config.data.transform.append({})
            vocab = getattr(import_module('torchtext.vocab'),
                            config.data.transform[0])(
                                dim=config.model.in_channels,
                                **config.data.transform[1])

            if 'classification' in config.model.task.lower():
                data = [
                    vocab.get_vecs_by_tokens(tokenizer(x),
                                             lower_case_backup=True)
                    for x in x_all
                ]
                targets = label_to_index(y_all)
            else:
                data = [
                    vocab.get_vecs_by_tokens(tokenizer(x),
                                             lower_case_backup=True)
                    for x in x_all
                ]
                targets = [
                    vocab.get_vecs_by_tokens(tokenizer(y),
                                             lower_case_backup=True)
                    for y in y_all
                ]
                targets = pad_sequence(targets).transpose(
                    0, 1)[:, :raw_args['max_len'], :]
            data = pad_sequence(data).transpose(0,
                                                1)[:, :raw_args['max_len'], :]
        # Split data to raw
        num_items = [len(ds) for ds in x_list]
        data_list, cnt = [], 0
        for num in num_items:
            data_list.append([
                (x, y)
                for x, y in zip(data[cnt:cnt + num], targets[cnt:cnt + num])
            ])
            cnt += num

        if len(data_list) == 3:
            # Use raw splits
            data_dict = {
                'train': data_list[0],
                'val': data_list[1],
                'test': data_list[2]
            }
        elif len(data_list) == 2:
            # Split train to (train, val)
            data_dict = {
                'train': data_list[0],
                'val': None,
                'test': data_list[1]
            }
            if splits:
                train_size = int(splits[0] * len(data_dict['train']))
                val_size = len(data_dict['train']) - train_size
                lengths = [train_size, val_size]
                data_dict['train'], data_dict[
                    'val'] = torch.utils.data.dataset.random_split(
                        data_dict['train'], lengths)
        else:
            # Use config.data.splits
            data_dict = {}
            train_size = int(splits[0] * len(data_list[0]))
            val_size = int(splits[1] * len(data_list[0]))
            test_size = len(data_list[0]) - train_size - val_size
            lengths = [train_size, val_size, test_size]
            data_dict['train'], data_dict['val'], data_dict[
                'test'] = torch.utils.data.dataset.random_split(
                    data_list[0], lengths)

        return data_dict

    def load_torchaudio_data(name, splits=None, config=None):
        import torchaudio

        # dataset_func = getattr(import_module('torchaudio.datasets'), name)
        raise NotImplementedError

    def load_torch_geometric_data(name, splits=None, config=None):
        import torch_geometric

        # dataset_func = getattr(import_module('torch_geometric.datasets'),
        # name)
        raise NotImplementedError

    def load_huggingface_datasets_data(name, splits=None, config=None):
        from datasets import load_dataset

        if config.data.args:
            raw_args = config.data.args[0]
        else:
            raw_args = {}
        assert 'max_len' in raw_args, "Miss key 'max_len' in " \
                                      "`config.data.args`."
        filtered_args = filter_dict(load_dataset, raw_args)
        dataset = load_dataset(path=config.data.root,
                               name=name,
                               **filtered_args)
        if config.model.type.endswith('transformers'):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.type.split('@')[0])

        for split in dataset:
            x_all = [i['sentence'] for i in dataset[split]]
            targets = [i['label'] for i in dataset[split]]

            x_all = tokenizer(x_all,
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=raw_args['max_len'])
            data = [{key: value[i]
                     for key, value in x_all.items()}
                    for i in range(len(next(iter(x_all.values()))))]
            dataset[split] = (data, targets)
        data_dict = {
            'train': [(x, y)
                      for x, y in zip(dataset['train'][0], dataset['train'][1])
                      ],
            'val': [(x, y) for x, y in zip(dataset['validation'][0],
                                           dataset['validation'][1])],
            'test': [
                (x, y) for x, y in zip(dataset['test'][0], dataset['test'][1])
            ] if (set(dataset['test'][1]) - set([-1])) else None,
        }
        return data_dict

    def load_openml_data(tid, splits=None, config=None):
        import openml
        from sklearn.model_selection import train_test_split

        task = openml.tasks.get_task(int(tid))
        did = task.dataset_id
        dataset = openml.datasets.get_dataset(did)
        data, targets, _, _ = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute)

        train_data, test_data, train_targets, test_targets = train_test_split(
            data, targets, train_size=splits[0], random_state=config.seed)
        val_data, test_data, val_targets, test_targets = train_test_split(
            test_data,
            test_targets,
            train_size=splits[1] / (1. - splits[0]),
            random_state=config.seed)
        data_dict = {
            'train': [(x, y) for x, y in zip(train_data, train_targets)],
            'val': [(x, y) for x, y in zip(val_data, val_targets)],
            'test': [(x, y) for x, y in zip(test_data, test_targets)]
        }
        return data_dict

    DATA_LOAD_FUNCS = {
        'torchvision': load_torchvision_data,
        'torchtext': load_torchtext_data,
        'torchaudio': load_torchaudio_data,
        'torch_geometric': load_torch_geometric_data,
        'huggingface_datasets': load_huggingface_datasets_data,
        'openml': load_openml_data
    }

    modified_config = config.clone()

    # Load dataset
    splits = modified_config.data.splits
    name, package = modified_config.data.type.split('@')

    dataset = DATA_LOAD_FUNCS[package.lower()](name, splits, modified_config)
    splitter = get_splitter(modified_config)

    data_local_dict = {
        x: {}
        for x in range(1, modified_config.federate.client_num + 1)
    }

    # Build dict of Dataloader
    train_label_distribution = None
    for split in dataset:
        if dataset[split] is None:
            continue
        train_labels = list()
        for i, ds in enumerate(
                splitter(dataset[split], prior=train_label_distribution)):
            labels = [x[1] for x in ds]
            if split == 'train':
                train_labels.append(labels)
                data_local_dict[i + 1][split] = DataLoader(
                    ds,
                    batch_size=modified_config.data.batch_size,
                    shuffle=True,
                    num_workers=modified_config.data.num_workers)
            else:
                data_local_dict[i + 1][split] = DataLoader(
                    ds,
                    batch_size=modified_config.data.batch_size,
                    shuffle=False,
                    num_workers=modified_config.data.num_workers)

        if modified_config.data.consistent_label_distribution and len(
                train_labels) > 0:
            train_label_distribution = train_labels

    return data_local_dict, modified_config


def get_data(config):
    """Instantiate the dataset and update the configuration accordingly if
    necessary.
    Arguments:
        config (obj): a cfg node object.
    Returns:
        obj: The dataset object.
        cfg.node: The updated configuration.
    """
    for func in register.data_dict.values():
        data_and_config = func(config)
        if data_and_config is not None:
            return data_and_config
    if config.data.type.lower() == 'toy':
        data, modified_config = load_toy_data(config)
    elif config.data.type.lower() == 'quadratic':
        from federatedscope.tabular.dataloader import load_quadratic_dataset
        data, modified_config = load_quadratic_dataset(config)
    elif config.data.type.lower() in ['femnist', 'celeba']:
        from federatedscope.cv.dataloader import load_cv_dataset
        data, modified_config = load_cv_dataset(config)
    elif config.data.type.lower() in [
            'shakespeare', 'twitter', 'subreddit', 'synthetic'
    ]:
        from federatedscope.nlp.dataloader import load_nlp_dataset
        data, modified_config = load_nlp_dataset(config)
    elif config.data.type.lower() in [
            'cora',
            'citeseer',
            'pubmed',
            'dblp_conf',
            'dblp_org',
    ] or config.data.type.lower().startswith('csbm'):
        from federatedscope.gfl.dataloader import load_nodelevel_dataset
        data, modified_config = load_nodelevel_dataset(config)
    elif config.data.type.lower() in ['ciao', 'epinions', 'fb15k-237', 'wn18']:
        from federatedscope.gfl.dataloader import load_linklevel_dataset
        data, modified_config = load_linklevel_dataset(config)
    elif config.data.type.lower() in [
            'hiv', 'proteins', 'imdb-binary', 'bbbp', 'tox21', 'bace', 'sider',
            'clintox', 'esol', 'freesolv', 'lipo'
    ] or config.data.type.startswith('graph_multi_domain'):
        from federatedscope.gfl.dataloader import load_graphlevel_dataset
        data, modified_config = load_graphlevel_dataset(config)
    elif config.data.type.lower() == 'vertical_fl_data':
        from federatedscope.vertical_fl.dataloader import load_vertical_data
        data, modified_config = load_vertical_data(config, generate=True)
    elif 'movielens' in config.data.type.lower():
        from federatedscope.mf.dataloader import load_mf_dataset
        data, modified_config = load_mf_dataset(config)
    elif '@' in config.data.type.lower():
        data, modified_config = load_external_data(config)
    elif 'cikmcup' in config.data.type.lower():
        from federatedscope.gfl.dataset.cikm_cup import load_cikmcup_data
        data, modified_config = load_cikmcup_data(config)
    else:
        raise ValueError('Data {} not found.'.format(config.data.type))

    if config.federate.mode.lower() == 'standalone':
        return data, modified_config
    else:
        # Invalid data_idx
        if config.distribute.data_idx not in data.keys():
            data_idx = np.random.choice(list(data.keys()))
            logger.warning(
                f"The provided data_idx={config.distribute.data_idx} is "
                f"invalid, so that we randomly sample a data_idx as {data_idx}"
            )
        else:
            data_idx = config.distribute.data_idx
        return data[data_idx], config


def merge_data(all_data, merged_max_data_id):
    dataset_names = list(all_data[1].keys())  # e.g., train, test, val
    import torch.utils.data
    assert len(dataset_names) >= 1, \
        "At least one sub-dataset is required in client 1"
    data_name = "test" if "test" in dataset_names else dataset_names[0]
    if isinstance(all_data[1][data_name], dict):
        data_elem_names = list(all_data[1][data_name].keys())  # e.g., x, y
        merged_data = {name: defaultdict(list) for name in dataset_names}
        for data_id in range(1, merged_max_data_id):
            for d_name in dataset_names:
                for elem_name in data_elem_names:
                    merged_data[d_name][elem_name].append(
                        all_data[data_id][d_name][elem_name])
        for d_name in dataset_names:
            for elem_name in data_elem_names:
                merged_data[d_name][elem_name] = np.concatenate(
                    merged_data[d_name][elem_name])
    elif issubclass(type(all_data[1][data_name]), torch.utils.data.DataLoader):
        merged_data = {name: all_data[1][name] for name in dataset_names}
        for data_id in range(2, merged_max_data_id):
            for d_name in dataset_names:
                merged_data[d_name].dataset.extend(
                    all_data[data_id][d_name].dataset)
    else:
        raise NotImplementedError(
            "Un-supported type when merging data across different clients."
            f"Your data type is {type(all_data[1][data_name])}. "
            f"Currently we only support the following forms: "
            " 1): {data_id: {train: {x:ndarray, y:ndarray}} }"
            " 2): {data_id: {train: DataLoader }")
    return merged_data
