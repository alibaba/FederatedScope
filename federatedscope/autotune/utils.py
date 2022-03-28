from copy import deepcopy
import math

import pandas as pd

from federatedscope.config import cfg
from federatedscope.autotune.choice_types import Continuous, Discrete


def split_raw_config(raw_dict):
    '''
    Arguments:
        raw_dict (dict): the loaded yaml file.
    Returns:
        det_config (dict): determined hyper-parameters.
        tbd_config (dict): to be determined hyper-parameters.
    '''
    det_config, tbd_config = dict(), dict()

    def traverse(root, rootname):
        if isinstance(root, dict):
            for k, v in root.items():
                traverse(v, rootname + '.' + k if rootname else k)
        else:
            if isinstance(root, Continuous) or isinstance(root, Discrete):
                tbd_config[rootname] = root
            else:
                det_config[rootname] = root

    traverse(raw_dict, '')
    return det_config, tbd_config


def generate_candidates(search_space):
    '''
    Arguments:
        search_space (dict): the search space.
    Returns:
        cands (list): each eleemnt is a dict corresponding to a specific configuration.
    '''

    # enumerate all combinations
    cands = []

    def traverse(it, idx, cur):
        if idx >= len(it):
            cands.append(deepcopy(cur))
            return
        k = it[idx]
        for val in search_space[k]:
            cur[k] = val
            traverse(it, idx + 1, cur)

    keys = [key for key in search_space]
    traverse(keys, 0, dict())

    return cands


def config2cmdargs(config):
    '''
    Arguments:
        config (dict): key is cfg node name, value is the choice of hyper-parameter.
    Returns:
        results (list): cmd args
    '''
    results = []
    for k, v in config.items():
        results.append(k)
        results.append(v)
    return results


def config2str(config):
    '''
    Arguments:
        config (dict): key is cfg node name, value is the choice of hyper-parameter.
    Returns:
        name (str): the string representation of this config
    '''
    vals = []
    for k in config:
        idx = k.rindex('.')
        vals.append(k[idx + 1:])
        vals.append(str(config[k]))
    name = '_'.join(vals)
    return name


def summarize_hpo_results(configs, perfs, white_list=None, desc=False):
    cols = [k for k in configs[0] if (white_list is None or k in white_list)
            ] + ['performance']
    d = [[
        trial_cfg[k]
        for k in trial_cfg if (white_list is None or k in white_list)
    ] + [result] for trial_cfg, result in zip(configs, perfs)]
    d = sorted(d, key=lambda ele: ele[-1], reverse=desc)
    df = pd.DataFrame(d, columns=cols)
    return df
