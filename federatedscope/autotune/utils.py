from copy import deepcopy
import math

import yaml
import pandas as pd
import ConfigSpace as CS

from federatedscope.autotune.choice_types import Continuous, Discrete


def parse_search_space(config_path):
    """Parse yaml format configuration to generate search space
    Arguments:
        config_path (str): the path of the yaml file.
    :returns: the search space.
    :rtype: ConfigSpace object
    """

    ss = CS.ConfigurationSpace()

    with open(config_path, 'r') as ips:
        raw_ss_config = yaml.load(ips, Loader=yaml.FullLoader)

    for k in raw_ss_config.keys():
        name = k
        v = raw_ss_config[k]
        hyper_type = v['type']
        del v['type']
        v['name'] = name

        if hyper_type == 'float':
            hyper_config = CS.UniformFloatHyperparameter(**v)
        elif hyper_type == 'int':
            hyper_config = CS.UniformIntegerHyperparameter(**v)
        elif hyper_type == 'cate':
            hyper_config = CS.CategoricalHyperparameter(**v)
        else:
            raise ValueError("Unsupported hyper type {}".format(hyper_type))
        ss.add_hyperparameter(hyper_config)

    return ss


def config2cmdargs(config):
    '''
    Arguments:
        config (dict): key is cfg node name, value is the specified value.
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
