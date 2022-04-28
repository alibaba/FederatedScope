import logging
import os
import random
import time
import math
from datetime import datetime
from os import path as osp
import ssl
import urllib.request

import numpy as np
# Blind torch
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


def update_logger(cfg, clear_before_add=False):
    import os
    import sys
    import logging

    root_logger = logging.getLogger("federatedscope")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        logging_fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(logging_fmt))
        root_logger.addHandler(handler)

    # update level
    if cfg.verbose > 0:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN
        logger.warning("Skip DEBUG/INFO messages")
    root_logger.setLevel(logging_level)

    # ================ create outdir to save log, exp_config, models, etc,.
    if cfg.outdir == "":
        cfg.outdir = os.path.join(os.getcwd(), "exp")
    cfg.outdir = os.path.join(cfg.outdir, cfg.expname)

    # if exist, make directory with given name and time
    if os.path.isdir(cfg.outdir) and os.path.exists(cfg.outdir):
        outdir = os.path.join(cfg.outdir, "sub_exp" +
                              datetime.now().strftime('_%Y%m%d%H%M%S')
                              )  # e.g., sub_exp_20220411030524
        while os.path.exists(outdir):
            time.sleep(1)
            outdir = os.path.join(
                cfg.outdir,
                "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
        cfg.outdir = outdir
    # if not, make directory with given name
    os.makedirs(cfg.outdir)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(cfg.outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    root_logger.addHandler(fh)
    #sys.stderr = sys.stdout

    root_logger.info(f"the output dir is {cfg.outdir}")


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


def filter_by_specified_keywords(param_name, config):
    '''
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    '''
    preserve = True
    for kw in config.personalization.local_param:
        if kw in param_name:
            preserve = False
            break
    return preserve


def get_random(type, sample_shape, params, device):
    if not hasattr(distributions, type):
        raise NotImplementedError("Distribution {} is not implemented, please refer to ```torch.distributions```" \
                                  "(https://pytorch.org/docs/stable/distributions.html).".format(type))
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

    Modified from `https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py`

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


def update_best_result(best_results,
                       results,
                       results_type,
                       round_wise_update_key="val_loss"):
    """
        update best evaluation results.
        by default, the update is based on validation loss with `round_wise_update_key="val_loss" `
    """
    update_best_this_round = False
    if not isinstance(results, dict):
        raise ValueError(
            f"update best results require `results` a dict, but got {type(results)}"
        )
    else:
        if results_type not in best_results:
            best_results[results_type] = dict()
        best_result = best_results[results_type]
        # update different keys separately: the best values can be in different rounds
        if round_wise_update_key is None:
            for key in results:
                cur_result = results[key]
                if 'loss' in key or 'std' in key:  # the smaller, the better
                    if results_type == "client_individual":
                        cur_result = min(cur_result)
                    if key not in best_result or cur_result < best_result[key]:
                        best_result[key] = cur_result
                        update_best_this_round = True

                elif 'acc' in key:  # the larger, the better
                    if results_type == "client_individual":
                        cur_result = max(cur_result)
                    if key not in best_result or cur_result > best_result[key]:
                        best_result[key] = cur_result
                        update_best_this_round = True
                else:
                    # unconcerned metric
                    pass
        # update different keys round-wise: if find better round_wise_update_key, update others at the same time
        else:
            if round_wise_update_key not in [
                    "val_loss", "val_acc", "val_std", "test_loss", "test_acc",
                    "test_std", "test_avg_loss", "loss"
            ]:
                raise NotImplementedError(
                    f"We currently support round_wise_update_key as one of "
                    f"['val_loss', 'val_acc', 'val_std', 'test_loss', 'test_acc', 'test_std'] "
                    f"for round-wise best results update, but got {round_wise_update_key}."
                )

            found_round_wise_update_key = False
            sorted_keys = []
            for key in results:
                if round_wise_update_key in key:
                    sorted_keys.insert(0, key)
                    found_round_wise_update_key = True
                else:
                    sorted_keys.append(key)
            if not found_round_wise_update_key:
                raise ValueError(
                    "The round_wise_update_key is not in target results, "
                    "use another key or check the name. \n"
                    f"Your round_wise_update_key={round_wise_update_key}, "
                    f"the keys of results are {list(results.keys())}")

            for key in sorted_keys:
                cur_result = results[key]
                if update_best_this_round or \
                        ('loss' in round_wise_update_key and 'loss' in key) or \
                        ('std' in round_wise_update_key and 'std' in key):
                    if results_type == "client_individual":
                        cur_result = min(cur_result)
                    if update_best_this_round or \
                            key not in best_result or cur_result < best_result[key]:
                        best_result[key] = cur_result
                        update_best_this_round = True
                elif update_best_this_round or \
                        'acc' in round_wise_update_key and 'acc' in key:
                    if results_type == "client_individual":
                        cur_result = max(cur_result)
                    if update_best_this_round or \
                            key not in best_result or cur_result > best_result[key]:
                        best_result[key] = cur_result
                        update_best_this_round = True
                else:
                    # unconcerned metric
                    pass

    if update_best_this_round:
        logging.info(f"Find new best result: {best_results}")
