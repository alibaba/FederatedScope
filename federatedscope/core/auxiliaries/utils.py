import collections
import json
import logging
import math
import os
import random
import signal
import ssl
import urllib.request
from collections import defaultdict
from os import path as osp
import pickle

import numpy as np

# Blind torch
import torch.utils

try:
    import torch
    import torchvision
    import torch.distributions as distributions
except ImportError:
    torch = None
    torchvision = None
    distributions = None

logger = logging.getLogger(__name__)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    else:
        import tensorflow as tf
        tf.set_random_seed(seed)


def get_dataset(type, root, transform, target_transform, download=True):
    if isinstance(type, str):
        if hasattr(torchvision.datasets, type):
            return getattr(torchvision.datasets,
                           type)(root=root,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
        else:
            raise NotImplementedError('Dataset {} not implement'.format(type))
    else:
        raise TypeError()


def save_local_data(dir_path,
                    train_data=None,
                    train_targets=None,
                    test_data=None,
                    test_targets=None,
                    val_data=None,
                    val_targets=None):
    r"""
    https://github.com/omarfoq/FedEM/blob/main/data/femnist/generate_data.py

    save (`train_data`, `train_targets`) in {dir_path}/train.pt,
    (`val_data`, `val_targets`) in {dir_path}/val.pt
    and (`test_data`, `test_targets`) in {dir_path}/test.pt
    :param dir_path:
    :param train_data:
    :param train_targets:
    :param test_data:
    :param test_targets:
    :param val_data:
    :param val_targets
    """
    if (train_data is not None) and (train_targets is not None):
        torch.save((train_data, train_targets), osp.join(dir_path, "train.pt"))

    if (test_data is not None) and (test_targets is not None):
        torch.save((test_data, test_targets), osp.join(dir_path, "test.pt"))

    if (val_data is not None) and (val_targets is not None):
        torch.save((val_data, val_targets), osp.join(dir_path, "val.pt"))


def filter_by_specified_keywords(param_name, filter_keywords):
    '''
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    '''
    preserve = True
    for kw in filter_keywords:
        if kw in param_name:
            preserve = False
            break
    return preserve


def get_random(type, sample_shape, params, device):
    if not hasattr(distributions, type):
        raise NotImplementedError("Distribution {} is not implemented, "
                                  "please refer to ```torch.distributions```"
                                  "(https://pytorch.org/docs/stable/ "
                                  "distributions.html).".format(type))
    generator = getattr(distributions, type)(**params)
    return generator.sample(sample_shape=sample_shape).to(device)


def batch_iter(data, batch_size=64, shuffled=True):
    assert 'x' in data and 'y' in data
    data_x = data['x']
    data_y = data['y']
    data_size = len(data_y)
    num_batches_per_epoch = math.ceil(data_size / batch_size)

    while True:
        shuffled_index = np.random.permutation(
            np.arange(data_size)) if shuffled else np.arange(data_size)
        for batch in range(num_batches_per_epoch):
            start_index = batch * batch_size
            end_index = min(data_size, (batch + 1) * batch_size)
            sample_index = shuffled_index[start_index:end_index]
            yield {'x': data_x[sample_index], 'y': data_y[sample_index]}


def merge_dict(dict1, dict2):
    # Merge results for history
    for key, value in dict2.items():
        if key not in dict1:
            if isinstance(value, dict):
                dict1[key] = merge_dict({}, value)
            else:
                dict1[key] = [value]
        else:
            if isinstance(value, dict):
                merge_dict(dict1[key], value)
            else:
                dict1[key].append(value)
    return dict1


def download_url(url: str, folder='folder'):
    r"""Downloads the content of an url to a folder.

    Modified from `https://github.com/pyg-team/pytorch_geometric/blob/master
    /torch_geometric/data/download.py`

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        path (string): File path of downloaded files.
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


def move_to(obj, device):
    import torch
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def param2tensor(param):
    import torch
    if isinstance(param, list):
        param = torch.FloatTensor(param)
    elif isinstance(param, int):
        param = torch.tensor(param, dtype=torch.long)
    elif isinstance(param, float):
        param = torch.tensor(param, dtype=torch.float)
    return param


class Timeout(object):
    def __init__(self, seconds, max_failure=5):
        self.seconds = seconds
        self.max_failure = max_failure

    def __enter__(self):
        def signal_handler(signum, frame):
            raise TimeoutError()

        if self.seconds > 0:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

    def reset(self):
        signal.alarm(self.seconds)

    def block(self):
        signal.alarm(0)

    def exceed_max_failure(self, num_failure):
        return num_failure > self.max_failure


def format_log_hooks(hooks_set):
    def format_dict(target_dict):
        print_dict = collections.defaultdict(list)
        for k, v in target_dict.items():
            for element in v:
                print_dict[k].append(element.__name__)
        return print_dict

    if isinstance(hooks_set, list):
        print_obj = [format_dict(_) for _ in hooks_set]
    elif isinstance(hooks_set, dict):
        print_obj = format_dict(hooks_set)
    return json.dumps(print_obj, indent=2).replace('\n', '\n\t')


def get_resource_info(filename):
    if filename is None or not os.path.exists(filename):
        logger.info('The device information file is not provided')
        return None

    # Users can develop this loading function according to resource_info_file
    # As an example, we use the device_info provided by FedScale (FedScale:
    # Benchmarking Model and System Performance of Federated Learning
    # at Scale), which can be downloaded from
    # https://github.com/SymbioticLab/FedScale/blob/master/benchmark/dataset/
    # data/device_info/client_device_capacity The expected format is
    # { INDEX:{'computation': FLOAT_VALUE_1, 'communication': FLOAT_VALUE_2}}
    with open(filename, 'br') as f:
        device_info = pickle.load(f)
    return device_info


def calculate_time_cost(instance_number,
                        comm_size,
                        comp_speed=None,
                        comm_bandwidth=None,
                        augmentation_factor=3.0):
    # Served as an example, this cost model is adapted from FedScale at
    # https://github.com/SymbioticLab/FedScale/blob/master/fedscale/core/
    # internal/client.py#L35 (Apache License Version 2.0)
    # Users can modify this function according to customized cost model
    if comp_speed is not None and comm_bandwidth is not None:
        comp_cost = augmentation_factor * instance_number * comp_speed
        comm_cost = 2.0 * comm_size / comm_bandwidth
    else:
        comp_cost = 0
        comm_cost = 0

    return comp_cost, comm_cost


def calculate_batch_epoch_num(steps, batch_or_epoch, num_data, batch_size,
                              drop_last):
    num_batch_per_epoch = num_data // batch_size + int(
        not drop_last and bool(num_data % batch_size))
    if num_batch_per_epoch == 0:
        raise RuntimeError(
            "The number of batch is 0, please check 'batch_size' or set "
            "'drop_last' as False")
    elif batch_or_epoch == "epoch":
        num_epoch = steps
        num_batch_last_epoch = num_batch_per_epoch
        num_total_batch = steps * num_batch_per_epoch
    else:
        num_epoch = math.ceil(steps / num_batch_per_epoch)
        num_batch_last_epoch = steps % num_batch_per_epoch or \
            num_batch_per_epoch
        num_total_batch = steps
    return num_batch_per_epoch, num_batch_last_epoch, num_epoch, \
        num_total_batch


def merge_param_dict(raw_param, filtered_param):
    for key in filtered_param.keys():
        raw_param[key] = filtered_param[key]
    return raw_param


def merge_data(all_data, merged_max_data_id, specified_dataset_name=None):
    if specified_dataset_name is None:
        dataset_names = list(all_data[1].keys())  # e.g., train, test, val
    else:
        if not isinstance(specified_dataset_name, list):
            specified_dataset_name = [specified_dataset_name]
        dataset_names = specified_dataset_name

    import torch.utils.data
    assert len(dataset_names) >= 1, \
        "At least one sub-dataset is required in client 1"
    data_name = "test" if "test" in dataset_names else dataset_names[0]
    id_has_key = 1
    while "test" not in all_data[id_has_key]:
        id_has_key += 1
        if len(all_data) <= id_has_key:
            raise KeyError(f'All data do not key {data_name}.')
    if isinstance(all_data[id_has_key][data_name], dict):
        data_elem_names = list(
            all_data[id_has_key][data_name].keys())  # e.g., x, y
        merged_data = {name: defaultdict(list) for name in dataset_names}
        for data_id in range(1, merged_max_data_id):
            for d_name in dataset_names:
                if d_name not in all_data[data_id]:
                    continue
                for elem_name in data_elem_names:
                    merged_data[d_name][elem_name].append(
                        all_data[data_id][d_name][elem_name])
        for d_name in dataset_names:
            for elem_name in data_elem_names:
                merged_data[d_name][elem_name] = np.concatenate(
                    merged_data[d_name][elem_name])
    elif issubclass(type(all_data[id_has_key][data_name]),
                    torch.utils.data.DataLoader):
        merged_data = {
            name: all_data[id_has_key][name]
            for name in dataset_names
        }
        for data_id in range(1, merged_max_data_id):
            if data_id == id_has_key:
                continue
            for d_name in dataset_names:
                if d_name not in all_data[data_id]:
                    continue
                merged_data[d_name].dataset.extend(
                    all_data[data_id][d_name].dataset)
    else:
        raise NotImplementedError(
            "Un-supported type when merging data across different clients."
            f"Your data type is {type(all_data[id_has_key][data_name])}. "
            f"Currently we only support the following forms: "
            " 1): {data_id: {train: {x:ndarray, y:ndarray}} }"
            " 2): {data_id: {train: DataLoader }")
    return merged_data
