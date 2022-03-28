import logging
import os
import os.path as osp
import random
import copy
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
import torch.distributions as distributions

import federatedscope.register as register


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(cfg):
    if cfg.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format=
            "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format=
            "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # ================ create outdir to save log, exp_config, models, etc,.

    if cfg.outdir == "":
        cfg.outdir = os.path.join(os.getcwd(), "exp")
    cfg.outdir = os.path.join(cfg.outdir, cfg.expname)

    # if exist, make directory with given name and time
    if os.path.isdir(cfg.outdir) and os.path.exists(cfg.outdir):
        # cfg.outdir = cfg.outdir + datetime.now().strftime('_%m-%d_%H:%M:%S')
        cfg.outdir = os.path.join(
            cfg.outdir, "sub_exp" + datetime.now().strftime('_%m-%d_%H:%M:%S'))
        while os.path.exists(cfg.outdir):
            time.sleep(1)
            cfg.outdir = os.path.join(
                cfg.outdir,
                "sub_exp" + datetime.now().strftime('_%m-%d_%H:%M:%S'))
    # if not, make directory with given name
    os.makedirs(cfg.outdir)

    logger = logging.getLogger()
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(cfg.outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    logger.addHandler(fh)
    sys.stderr = sys.stdout


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


def formatted_logging(results,
                      rnd,
                      role=-1,
                      forms=['weighted_avg', 'avg', 'fairness', 'raw']):
    # fomatted the output
    output = {'Role': role, 'Round': rnd}
    for form in forms:
        new_results = copy.deepcopy(results)
        if not role.lower().startswith('server') or form == 'raw':
            output['Results_raw'] = new_results
        elif form not in SUPPORTED_FORMS:
            continue
        else:
            for mode in ['train', 'val', 'test']:
                if f'{mode}_total' not in results:
                    continue
                num = np.array(new_results[f'{mode}_total'])
                for key in results.keys():
                    if key in [f'{mode}_total', f'{mode}_correct']:
                        new_results[key] = np.mean(new_results[key])
                    else:
                        if form == 'weighted_avg':
                            new_results[key] = np.sum(
                                np.array(new_results[key]) * num) / np.sum(num)
                        if form == "avg":
                            new_results[key] = np.mean(new_results[key])
                        if form == "fairness":
                            # by default, log the std and decile
                            all_res = copy.copy(results[key])
                            new_results.pop(
                                key, None)  # delete the redundant original one
                            all_res.sort()
                            new_results[f"{key}_std"] = np.std(
                                np.array(all_res))
                            new_results[f"{key}_bottom_decile"] = all_res[
                                len(all_res) // 10]
                            new_results[f"{key}_top_decile"] = all_res[
                                len(all_res) * 9 // 10]
            output[f'Results_{form}'] = new_results

    return output


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
