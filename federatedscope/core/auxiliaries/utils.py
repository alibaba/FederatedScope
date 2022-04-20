import logging
import os
import os.path as osp
import random
import copy
import sys
import ssl
import time
import math
import urllib.request
from datetime import datetime

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

import federatedscope.register as register

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
        handler.setFormatter(
            logging.Formatter(logging_fmt))
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


def merge_local_results(local_results, aggr):
    '''
    Arguments:
        local_results (list): each element is a dict of client-specific performances.
        aggr (str): 'mean' or 'sum'.
    Returns:
        aggr_results (dict): same keys as input.
    '''

    aggr_results = copy.deepcopy(local_results[0])
    for i in range(1, len(local_results)):
        for k, v in local_results[i].items():
            aggr_results[k] += v
    if aggr == 'mean':
        for k in aggr_results:
            aggr_results[k] /= len(local_results)
    return aggr_results


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


# Remove this
def calc_measurements(raw_results, config):
    '''
    Arguments:
        results (dict): metrics dict.
    
    Exmaple of all different key-value pairs:
            {
                'train_correct': 102.0,
                'train_loss': 103.47270369529724,
                'train_total': 150.0,
                'val_correct': 12.0,
                'val_loss': 13.088095366954803,
                'val_total': 19.0,
                'test_correct': 11.0,
                'test_loss': 13.129692792892456,
                'test_total': 19.0,
                'test_hits@5': 0.90,
                'roc_auc_score': 0.70,
                'best_valid_round': 2
            }
    '''

    # Logging output to logs
    logs = {}

    for mode in ['train', 'test', 'val']:
        if '{}_correct'.format(mode) in raw_results.keys():
            logs['{}_acc'.format(mode)] = raw_results['{}_correct'.format(
                mode)] / raw_results['{}_total'.format(mode)]

        if '{}_loss'.format(mode) in raw_results.keys():
            logs['{}_loss'.format(mode)] = raw_results['{}_loss'.format(
                mode)] / raw_results['{}_total'.format(mode)]
        elif 'avg_{}_loss'.format(mode) in raw_results.keys():
            logs['{}_loss'.format(mode)] = raw_results['avg_{}_loss'.format(
                mode)]

    hits_list = []
    for hit in config.eval.hits:
        logs[f'test_hits@{hit}'] = raw_results[
            f'test_hits@{hit}'] / raw_results['test_total']
    if config.eval.roc_auc:
        logs['roc_auc_score'] = raw_results['roc_auc_score']

    if 'best_valid_round' in raw_results:
        logs['best_valid_round'] = raw_results['best_valid_round']

    if 'B_val' in raw_results:
        logs['B_val'] = raw_results['B_val']

    return logs


SUPPORTED_FORMS = ['weighted_avg', 'avg', 'fairness', 'raw']


def formatted_logging(results, rnd, role=-1, forms=None):
    """
        format the output

        Args:
            results (dict): a dict to store the evaluation results {metric: value}
            rnd (int|string): FL round
            role (int|string): the output role
            forms (list): format type

        Returns:
            round_formatted_results (dict): a formatted results with different forms and roles,
            e.g.,
            {
            'Role': 'Server #',
            'Round': 200,
            'Results_weighted_avg': {
                'test_avg_loss': 0.58, 'test_acc': 0.67, 'test_correct': 3356, 'test_loss': 2892, 'test_total': 5000
                },
            'Results_avg': {
                'test_avg_loss': 0.57, 'test_acc': 0.67, 'test_correct': 3356, 'test_loss': 2892, 'test_total': 5000
                },
            'Results_fairness': {
                'test_correct': 3356,      'test_total': 5000,
                'test_avg_loss_std': 0.04, 'test_avg_loss_bottom_decile': 0.52, 'test_avg_loss_top_decile': 0.64,
                'test_acc_std': 0.06,      'test_acc_bottom_decile': 0.60,      'test_acc_top_decile': 0.75,
                'test_loss_std': 214.17,   'test_loss_bottom_decile': 2644.64,  'test_loss_top_decile': 3241.23
                },
            }
    """
    # TODO: better visualization via wandb or tensorboard
    # TODO: save the results logging into outdir/results.log
    if forms is None:
        forms = ['weighted_avg', 'avg', 'fairness', 'raw']
    round_formatted_results = {'Role': role, 'Round': rnd}
    for form in forms:
        new_results = copy.deepcopy(results)
        if not role.lower().startswith('server') or form == 'raw':
            round_formatted_results['Results_raw'] = new_results
        elif form not in SUPPORTED_FORMS:
            continue
        else:
            for key in results.keys():
                dataset_name = key.split("_")[0]
                if f'{dataset_name}_total' not in results:
                    raise ValueError(
                        "Results to be formatted should be include the dataset_num in the dict,"
                        f"with key = {dataset_name}_total")
                else:
                    dataset_num = np.array(results[f'{dataset_name}_total'])
                    if key in [
                            f'{dataset_name}_total', f'{dataset_name}_correct'
                    ]:
                        new_results[key] = np.mean(new_results[key])

                if key in [f'{dataset_name}_total', f'{dataset_name}_correct']:
                    new_results[key] = np.mean(new_results[key])
                else:
                    all_res = np.array(copy.copy(results[key]))
                    if form == 'weighted_avg':
                        new_results[key] = np.sum(
                            np.array(new_results[key]) *
                            dataset_num) / np.sum(dataset_num)
                    if form == "avg":
                        new_results[key] = np.mean(new_results[key])
                    if form == "fairness" and all_res.size > 1:
                        # by default, log the std and decile
                        new_results.pop(
                            key, None)  # delete the redundant original one
                        all_res.sort()
                        new_results[f"{key}_std"] = np.std(np.array(all_res))
                        new_results[f"{key}_bottom_decile"] = all_res[
                            all_res.size // 10]
                        new_results[f"{key}_top_decile"] = all_res[all_res.size
                                                                   * 9 // 10]
            round_formatted_results[f'Results_{form}'] = new_results

    return round_formatted_results


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
