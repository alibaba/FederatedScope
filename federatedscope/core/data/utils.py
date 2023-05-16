import copy
import inspect
import logging
import os
import re
import ssl
import urllib.request

import numpy as np
import os.path as osp

from random import shuffle
from collections import defaultdict

logger = logging.getLogger(__name__)


class RegexInverseMap:
    def __init__(self, n_dic, val):
        self._items = {}
        for key, values in n_dic.items():
            for value in values:
                self._items[value] = key
        self.__val = val

    def __getitem__(self, key):
        for regex in self._items.keys():
            if re.compile(regex).match(key):
                return self._items[regex]
        return self.__val

    def __repr__(self):
        return str(self._items.items())


def load_dataset(config, client_cfgs=None):
    """
    Loads the dataset for the given config from branches

    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Note:
        See https://federatedscope.io/docs/datazoo/ for all available data.
    """

    if config.data.type.lower() == 'toy':
        from federatedscope.tabular.dataloader.toy import load_toy_data
        dataset, modified_config = load_toy_data(config)
    elif config.data.type.lower() == 'quadratic':
        from federatedscope.tabular.dataloader import load_quadratic_dataset
        dataset, modified_config = load_quadratic_dataset(config)
    elif config.data.type.lower() in ['femnist', 'celeba']:
        from federatedscope.cv.dataloader import load_cv_dataset
        dataset, modified_config = load_cv_dataset(config)
    elif config.data.type.lower() in ['cifar4cl', 'cifar4lp']:
        from federatedscope.cl.dataloader import load_cifar_dataset
        dataset, modified_config = load_cifar_dataset(config)
    elif config.data.type.lower() in [
            'shakespeare', 'twitter', 'subreddit', 'synthetic'
    ]:
        from federatedscope.nlp.dataloader import load_nlp_dataset
        dataset, modified_config = load_nlp_dataset(config)
    elif config.data.type.lower() in [
            'cora',
            'citeseer',
            'pubmed',
            'dblp_conf',
            'dblp_org',
    ] or config.data.type.lower().startswith('csbm'):
        from federatedscope.gfl.dataloader import load_nodelevel_dataset
        dataset, modified_config = load_nodelevel_dataset(config)
    elif config.data.type.lower() in ['ciao', 'epinions', 'fb15k-237', 'wn18']:
        from federatedscope.gfl.dataloader import load_linklevel_dataset
        dataset, modified_config = load_linklevel_dataset(config)
    elif config.data.type.lower() in [
            'hiv', 'proteins', 'imdb-binary', 'bbbp', 'tox21', 'bace', 'sider',
            'clintox', 'esol', 'freesolv', 'lipo', 'cikmcup'
    ] or config.data.type.startswith('graph_multi_domain'):
        from federatedscope.gfl.dataloader import load_graphlevel_dataset
        dataset, modified_config = load_graphlevel_dataset(config)
    elif config.data.type.lower() in [
            'synthetic_vfl_data', 'adult', 'abalone', 'credit', 'blog'
    ]:
        from federatedscope.vertical_fl.dataloader import load_vertical_data
        generate = config.data.type.lower() == 'synthetic_vfl_data'
        dataset, modified_config = load_vertical_data(config,
                                                      generate=generate)
    elif 'movielens' in config.data.type.lower(
    ) or 'netflix' in config.data.type.lower():
        from federatedscope.mf.dataloader import load_mf_dataset
        dataset, modified_config = load_mf_dataset(config)
    elif 'hetero_nlp_tasks' in config.data.type.lower():
        from federatedscope.nlp.hetero_tasks.dataloader import \
            load_heteroNLP_data
        dataset, modified_config = load_heteroNLP_data(config, client_cfgs)
    elif '@llm' in config.data.type.lower():
        from federatedscope.llm.dataloader import load_llm_dataset
        dataset, modified_config = load_llm_dataset(config)
    elif '@' in config.data.type.lower():
        from federatedscope.core.data.utils import load_external_data
        dataset, modified_config = load_external_data(config)
    elif config.data.type is None or config.data.type == "":
        # The participant (only for server in this version) does not own data
        dataset = None
        modified_config = config
    else:
        raise ValueError('Dataset {} not found.'.format(config.data.type))
    return dataset, modified_config


def load_external_data(config=None):
    """
    Based on the configuration file, this function imports external \
    datasets and applies train/valid/test.

    Args:
        config: `CN` from `federatedscope/core/configs/config.py`

    Returns:
        (data, modified_config): tuple of ML split dataset, \
        and `CN` from `federatedscope/core/configs/config.py`, \
        which might be modified in the function.
    """

    import torch
    from importlib import import_module
    from torch.utils.data import DataLoader
    from federatedscope.core.auxiliaries.transform_builder import get_transform

    def load_torchvision_data(name, splits=None, config=None):
        from torch.utils.data import Subset

        dataset_func = getattr(import_module('torchvision.datasets'), name)
        transform_funcs, val_transform_funcs, test_transform_funcs = \
            get_transform(config, 'torchvision')
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
                                        **test_transform_funcs)
            if splits:
                dataset_val = dataset_func(root=config.data.root,
                                           train=True,
                                           **filtered_args,
                                           **val_transform_funcs)

                index = [i for i in range(len(dataset_train))]
                np.random.shuffle(index)
                train_size = int(splits[0] * len(dataset_train))
                train_idx_slice, val_idx_slice = index[:train_size], \
                    index[train_size:]

                dataset_train = Subset(dataset_train, train_idx_slice)
                dataset_val = Subset(dataset_val, val_idx_slice)

        elif 'split' in func_args:
            # Use raw split
            dataset_train = dataset_func(root=config.data.root,
                                         split='train',
                                         **filtered_args,
                                         **transform_funcs)
            dataset_val = dataset_func(root=config.data.root,
                                       split='valid',
                                       **filtered_args,
                                       **val_transform_funcs)
            dataset_test = dataset_func(root=config.data.root,
                                        split='test',
                                        **filtered_args,
                                        **test_transform_funcs)
        elif 'classes' in func_args:
            # Use raw split
            dataset_train = dataset_func(root=config.data.root,
                                         classes='train',
                                         **filtered_args,
                                         **transform_funcs)
            dataset_val = dataset_func(root=config.data.root,
                                       classes='valid',
                                       **filtered_args,
                                       **val_transform_funcs)
            dataset_test = dataset_func(root=config.data.root,
                                        classes='test',
                                        **filtered_args,
                                        **test_transform_funcs)
        else:
            # Use config.data.splits
            dataset_train = dataset_func(root=config.data.root,
                                         **filtered_args,
                                         **transform_funcs)
            dataset_val = dataset_func(root=config.data.root,
                                       **filtered_args,
                                       **val_transform_funcs)
            dataset_test = dataset_func(root=config.data.root,
                                        **filtered_args,
                                        **test_transform_funcs)

            index = [i for i in range(len(dataset_train))]
            np.random.shuffle(index)

            train_size = int(splits[0] * len(dataset_train))
            val_size = int(splits[1] * len(dataset_train))

            train_idx_slice = index[:train_size]
            val_idx_slice = index[train_size:train_size + val_size]
            test_idx_slice = index[train_size + val_size:]

            dataset_train = Subset(dataset_train, train_idx_slice)
            dataset_val = Subset(dataset_val, val_idx_slice)
            dataset_test = Subset(dataset_test, test_idx_slice)

        data_split_dict = {
            'train': dataset_train,
            'val': dataset_val,
            'test': dataset_test
        }

        return data_split_dict

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
            cache_path = os.path.join(os.getcwd(), "huggingface")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model.type.split('@')[0],
                    local_files_only=True,
                    cache_dir=cache_path)
            except Exception as e:
                logging.error(f"When loading cached file form "
                              f"{cache_path}, we faced the exception: \n "
                              f"{str(e)}")

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
            data_split_dict = {
                'train': data_list[0],
                'val': data_list[1],
                'test': data_list[2]
            }
        elif len(data_list) == 2:
            # Split train to (train, val)
            data_split_dict = {
                'train': data_list[0],
                'val': None,
                'test': data_list[1]
            }
            if splits:
                train_size = int(splits[0] * len(data_split_dict['train']))
                val_size = len(data_split_dict['train']) - train_size
                lengths = [train_size, val_size]
                data_split_dict['train'], data_split_dict[
                    'val'] = torch.utils.data.dataset.random_split(
                        data_split_dict['train'], lengths)
        else:
            # Use config.data.splits
            data_split_dict = {}
            train_size = int(splits[0] * len(data_list[0]))
            val_size = int(splits[1] * len(data_list[0]))
            test_size = len(data_list[0]) - train_size - val_size
            lengths = [train_size, val_size, test_size]
            data_split_dict['train'], data_split_dict['val'], data_split_dict[
                'test'] = torch.utils.data.dataset.random_split(
                    data_list[0], lengths)

        return data_split_dict

    def load_torchaudio_data(name, splits=None, config=None):

        # dataset_func = getattr(import_module('torchaudio.datasets'), name)
        raise NotImplementedError

    def load_huggingface_datasets_data(name, splits=None, config=None):
        import datasets
        from datasets import load_from_disk

        if config.data.args:
            raw_args = config.data.args[0]
        else:
            raw_args = {}
        assert 'max_len' in raw_args, "Miss key 'max_len' in " \
                                      "`config.data.args`."
        filtered_args = filter_dict(datasets.load_dataset, raw_args)
        logger.info("Begin to load huggingface dataset")
        if "hg_cache_dir" in raw_args:
            hugging_face_path = raw_args["hg_cache_dir"]
        else:
            hugging_face_path = os.getcwd()

        if "load_disk_dir" in raw_args:
            load_path = raw_args["load_disk_dir"]
            try:
                dataset = load_from_disk(load_path)
            except Exception as e:
                logging.error(f"When loading cached dataset form "
                              f"{load_path}, we faced the exception: \n "
                              f"{str(e)}")
        else:
            dataset = datasets.load_dataset(path=config.data.root,
                                            name=name,
                                            **filtered_args)
        if config.model.type.endswith('transformers'):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from transformers import AutoTokenizer
            logger.info("To load huggingface tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.type.split('@')[0],
                local_files_only=True,
                cache_dir=os.path.join(hugging_face_path, "transformers"))

        for split in dataset:
            x_all = [i['sentence'] for i in dataset[split]]
            targets = [i['label'] for i in dataset[split]]

            if split == "train" and "used_train_ratio" in raw_args and \
                    1 > raw_args['used_train_ratio'] > 0:
                selected_idx = [i for i in range(len(dataset[split]))]
                shuffle(selected_idx)
                selected_idx = selected_idx[:int(
                    len(selected_idx) * raw_args['used_train_ratio'])]
                x_all = [
                    element for i, element in enumerate(x_all)
                    if i in selected_idx
                ]
                targets = [
                    element for i, element in enumerate(targets)
                    if i in selected_idx
                ]

            x_all = tokenizer(x_all,
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=raw_args['max_len'])
            data = [{key: value[i]
                     for key, value in x_all.items()}
                    for i in range(len(next(iter(x_all.values()))))]
            dataset[split] = (data, targets)
        data_split_dict = {
            'train': [(x, y)
                      for x, y in zip(dataset['train'][0], dataset['train'][1])
                      ],
            'val': [(x, y) for x, y in zip(dataset['validation'][0],
                                           dataset['validation'][1])],
            'test': [
                (x, y) for x, y in zip(dataset['test'][0], dataset['test'][1])
            ] if (set(dataset['test'][1]) - set([-1])) else None,
        }
        original_train_size = len(data_split_dict["train"])

        if "half_val_dummy_test" in raw_args and raw_args[
                "half_val_dummy_test"]:
            # since the "test" set from GLUE dataset may be masked, we need to
            # submit to get the ground-truth, for fast FL experiments,
            # we split the validation set into two parts with the same size as
            # new test/val data
            original_val = [(x, y) for x, y in zip(dataset['validation'][0],
                                                   dataset['validation'][1])]
            data_split_dict["val"], data_split_dict[
                "test"] = original_val[:len(original_val) //
                                       2], original_val[len(original_val) //
                                                        2:]
        if "val_as_dummy_test" in raw_args and raw_args["val_as_dummy_test"]:
            # use the validation set as tmp test set,
            # and partial training set as validation set
            data_split_dict["test"] = data_split_dict["val"]
            data_split_dict["val"] = []
        if "part_train_dummy_val" in raw_args and 1 > raw_args[
                "part_train_dummy_val"] > 0:
            new_val_part = int(original_train_size *
                               raw_args["part_train_dummy_val"])
            data_split_dict["val"].extend(
                data_split_dict["train"][:new_val_part])
            data_split_dict["train"] = data_split_dict["train"][new_val_part:]
        if "part_train_dummy_test" in raw_args and 1 > raw_args[
                "part_train_dummy_test"] > 0:
            new_test_part = int(original_train_size *
                                raw_args["part_train_dummy_test"])
            data_split_dict["test"] = data_split_dict["val"]
            if data_split_dict["test"] is not None:
                data_split_dict["test"].extend(
                    data_split_dict["train"][:new_test_part])
            else:
                data_split_dict["test"] = (
                    data_split_dict["train"][:new_test_part])
            data_split_dict["train"] = data_split_dict["train"][new_test_part:]

        return data_split_dict

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
        data_split_dict = {
            'train': [(x, y) for x, y in zip(train_data, train_targets)],
            'val': [(x, y) for x, y in zip(val_data, val_targets)],
            'test': [(x, y) for x, y in zip(test_data, test_targets)]
        }
        return data_split_dict

    DATA_LOAD_FUNCS = {
        'torchvision': load_torchvision_data,
        'torchtext': load_torchtext_data,
        'torchaudio': load_torchaudio_data,
        'huggingface_datasets': load_huggingface_datasets_data,
        'openml': load_openml_data
    }

    modified_config = config.clone()

    # Load dataset
    splits = modified_config.data.splits
    name, package = modified_config.data.type.split('@')

    # Comply with the original train/val/test
    dataset = DATA_LOAD_FUNCS[package.lower()](name, splits, modified_config)
    data_split_tuple = (dataset.get('train'), dataset.get('val'),
                        dataset.get('test'))

    return data_split_tuple, modified_config


def convert_data_mode(data, config):
    """
    Convert ``StandaloneDataDict`` to ``ClientData`` in ``distributed`` mode.

    Args:
        data: ``StandaloneDataDict``
        config: configuration of FL course, see `federatedscope.core.configs`

    Returns:
        ``StandaloneDataDict`` in ``standalone`` mode, or ``ClientData`` in \
        ``distributed`` mode.
    """
    if config.federate.mode.lower() == 'standalone':
        return data
    else:
        # Invalid data_idx
        if config.distribute.data_idx == -1:
            return data
        elif config.distribute.data_idx not in data.keys():
            data_idx = np.random.choice(list(data.keys()))
            logger.warning(
                f"The provided data_idx={config.distribute.data_idx} is "
                f"invalid, so that we randomly sample a data_idx as {data_idx}"
            )
        else:
            data_idx = config.distribute.data_idx
        return data[data_idx]


def get_func_args(func):
    """
    Get the set of arguments that the function expects.

    Args:
        func: function to be analysis

    Returns:
        Arguments  that the function expects
    """
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    return sign


def filter_dict(func, kwarg):
    """
    Filters out the common keys of kwarg that are not in kwarg.

    Args:
        func: function to be filtered
        kwarg: dict to filter

    Returns:
        Filtered dict of arguments of the function.
    """
    sign = get_func_args(func)
    common_args = sign.intersection(kwarg.keys())
    filtered_dict = {key: kwarg[key] for key in common_args}
    return filtered_dict


def merge_data(all_data, merged_max_data_id=None, specified_dataset_name=None):
    """
    Merge data from client 1 to ``merged_max_data_id`` contained in given \
    ``all_data``.

    Args:
        all_data: ``StandaloneDataDict``
        merged_max_data_id: max merged data index
        specified_dataset_name: split name to be merged

    Returns:
        Merged data.
    """
    import torch.utils.data
    from federatedscope.core.data.wrap_dataset import WrapDataset

    # Assert
    if merged_max_data_id is None:
        merged_max_data_id = len(all_data) - 1
    assert merged_max_data_id >= 1
    if specified_dataset_name is None:
        dataset_names = list(all_data[1].keys())  # e.g., train, test, val
    else:
        if not isinstance(specified_dataset_name, list):
            specified_dataset_name = [specified_dataset_name]
        dataset_names = specified_dataset_name
    assert len(dataset_names) >= 1, \
        "At least one sub-dataset is required in client 1"

    data_name = "test" if "test" in dataset_names else dataset_names[0]
    id_contain_all_dataset_key = -1
    # check the existence of the data to be merged
    for client_id in range(1, merged_max_data_id + 1):
        contain_all_dataset_key = True
        for dataset_name in dataset_names:
            if dataset_name not in all_data[client_id]:
                contain_all_dataset_key = False
                logger.warning(f'Client {client_id} does not contain '
                               f'dataset key {dataset_name}.')
        if id_contain_all_dataset_key == -1 and contain_all_dataset_key:
            id_contain_all_dataset_key = client_id
    assert id_contain_all_dataset_key != -1, \
        "At least one client within [1, merged_max_data_id] should contain " \
        "all the key for expected dataset names."

    if issubclass(type(all_data[id_contain_all_dataset_key][data_name]),
                  torch.utils.data.DataLoader):
        if isinstance(all_data[id_contain_all_dataset_key][data_name].dataset,
                      WrapDataset):
            # e.g., x, y
            data_elem_names = list(all_data[id_contain_all_dataset_key]
                                   [data_name].dataset.dataset.keys())
            merged_data = {name: defaultdict(list) for name in dataset_names}
            for data_id in range(1, merged_max_data_id + 1):
                for d_name in dataset_names:
                    if d_name not in all_data[data_id]:
                        continue
                    for elem_name in data_elem_names:
                        merged_data[d_name][elem_name].append(
                            all_data[data_id]
                            [d_name].dataset.dataset[elem_name])
            for d_name in dataset_names:
                for elem_name in data_elem_names:
                    merged_data[d_name][elem_name] = np.concatenate(
                        merged_data[d_name][elem_name])
                merged_data[d_name] = WrapDataset(merged_data[d_name])
        else:
            client_data = {
                key: []
                for key in all_data[id_contain_all_dataset_key].keys()
            }
            for data_id in range(1, merged_max_data_id + 1):
                for d_name in dataset_names:
                    if d_name not in all_data[data_id]:
                        continue
                    else:
                        client_data[d_name].append(
                            all_data[data_id][d_name].dataset)
            merged_data = {
                key: torch.utils.data.ConcatDataset(client_data[key])
                for key in dataset_names
            }
    else:
        raise NotImplementedError(
            "Un-supported type when merging data across different clients."
            f"Your data type is "
            f"{type(all_data[id_contain_all_dataset_key][data_name])}. "
            f"Currently we only support the following forms: "
            " 1): {data_id: {train: {x:ndarray, y:ndarray}} }"
            " 2): {data_id: {train: DataLoader }")
    return merged_data


def save_local_data(dir_path,
                    train_data=None,
                    train_targets=None,
                    test_data=None,
                    test_targets=None,
                    val_data=None,
                    val_targets=None):
    r"""
    Save data to disk. Source: \
    https://github.com/omarfoq/FedEM/blob/main/data/femnist/generate_data.py

    Args:
        train_data: x of train data
        train_targets: y of train data
        test_data: x of test data
        test_targets: y of test data
        val_data: x of validation data
        val_targets:y of validation data

    Note:
        save ``(`train_data`, `train_targets`)`` in ``{dir_path}/train.pt``, \
        ``(`val_data`, `val_targets`)`` in ``{dir_path}/val.pt`` \
        and ``(`test_data`, `test_targets`)`` in ``{dir_path}/test.pt``
    """
    import torch
    if (train_data is not None) and (train_targets is not None):
        torch.save((train_data, train_targets), osp.join(dir_path, "train.pt"))

    if (test_data is not None) and (test_targets is not None):
        torch.save((test_data, test_targets), osp.join(dir_path, "test.pt"))

    if (val_data is not None) and (val_targets is not None):
        torch.save((val_data, val_targets), osp.join(dir_path, "val.pt"))


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        logger.info(f'File {file} exists, use existing file.')
        return path

    logger.info(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path
