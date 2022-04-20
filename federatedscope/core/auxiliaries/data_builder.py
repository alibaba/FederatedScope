import numpy as np
import pickle

import federatedscope.register as register


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
    import torch
    import inspect
    import logging
    from importlib import import_module
    from torch.utils.data import DataLoader
    from federatedscope.core.splitters import get_splitter

    def get_func_args(func):
        sign = inspect.signature(func).parameters.values()
        sign = set([val.name for val in sign])
        return sign

    def filter_dict(func, kwarg):
        sign = get_func_args(func)
        common_args = sign.intersection(kwarg.keys())
        filtered_dict = {key: kwarg[key] for key in common_args}
        return filtered_dict

    def build_transforms(T_names, transforms):
        transform_funcs = {}
        for name in T_names:
            if config.data[name]:
                # return composed transform or return list
                try:
                    transform_funcs[name] = transforms.Compose(
                        eval(config.data[name]))
                except:
                    transform_funcs[name] = eval(config.data[name])
            else:
                transform_funcs[name] = None
        return transform_funcs

    def load_torchvision_data(name, splits=None, config=None):
        import torchvision

        dataset_func = getattr(import_module('torchvision.datasets'), name)
        transforms = getattr(import_module('torchvision'), 'transforms')
        transform_funcs = build_transforms(['transform', 'target_transform'],
                                           transforms)
        transform_funcs = filter_dict(dataset_func.__init__, transform_funcs)
        raw_args = eval(config.data.args)
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
                dataset_train, dataset_val = torch.utils.data.dataset.random_split(
                    dataset_train, lengths)

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
            dataset_train, dataset_val, dataset_test = torch.utils.data.dataset.random_split(
                dataset, lengths)

        data_dict = {
            'train': dataset_train,
            'val': dataset_val,
            'test': dataset_test
        }

        return data_dict

    def load_torchtext_data(name, splits=None, config=None):
        import torchtext

        dataset_func = getattr(import_module('torchtext.datasets'), name)
        raise NotImplementedError

    def load_torchaudio_data(name, splits=None, config=None):
        import torchaudio

        dataset_func = getattr(import_module('torchaudio.datasets'), name)
        raise NotImplementedError

    def load_torch_geometric_data(name, splits=None, config=None):
        import torch_geometric

        dataset_func = getattr(import_module('torch_geometric.datasets'), name)
        raise NotImplementedError

    load_data = {
        'torchvision': load_torchvision_data,
        'torchtext': load_torchtext_data,
        'torchaudio': load_torchaudio_data,
        'torch_geometric': load_torch_geometric_data
    }

    # Load dataset
    splits = config.data.splits
    name, package = config.data.type.split('@')

    logging.info('Loading external dataset...')
    dataset = load_data[package.lower()](name, splits, config)
    splitter = get_splitter(config)

    data_local_dict = {x: {} for x in range(1, config.federate.client_num + 1)}

    # Build dict of Dataloader
    for split in dataset:
        if dataset[split] is None:
            continue
        for i, ds in enumerate(splitter(dataset[split])):
            if split == 'train':
                data_local_dict[i + 1][split] = DataLoader(
                    ds,
                    batch_size=config.data.batch_size,
                    shuffle=True,
                    num_workers=config.data.num_workers)
            else:
                data_local_dict[i + 1][split] = DataLoader(
                    ds,
                    batch_size=config.data.batch_size,
                    shuffle=False,
                    num_workers=config.data.num_workers)

    return data_local_dict, config


def get_data(config):
    for func in register.data_dict.values():
        data_and_config = func(config)
        if data_and_config is not None:
            return data_and_config
    if config.data.type.lower() == 'toy':
        data, modified_config = load_toy_data(config)
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
            'hiv', 'proteins', 'imdb-binary'
    ] or config.data.type.startswith('graph_multi_domain'):
        from federatedscope.gfl.dataloader import load_graphlevel_dataset
        data, modified_config = load_graphlevel_dataset(config)
    elif config.data.type.lower() == 'vertical_fl_data':
        from federatedscope.vertical_fl.dataloader import load_vertical_data
        data, modified_config = load_vertical_data(config, generate=True)
    elif 'movielens' in config.data.type.lower():
        from federatedscope.mf.dataloader import load_mf_dataset
        data, modified_config = load_mf_dataset(config)
    else:
        # Try to import external data
        try:
            data, modified_config = load_external_data(config)
        except:
            raise ValueError('Data {} not found.'.format(config.data.type))

    return data, modified_config
