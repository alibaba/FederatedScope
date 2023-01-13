import os.path

import yaml
import logging
import pandas as pd
import numpy as np
import ConfigSpace as CS

from federatedscope.core.configs.yacs_config import CfgNode

logger = logging.getLogger(__name__)


def generate_hpo_exp_name(cfg):
    return f'{cfg.hpo.scheduler}_{cfg.hpo.sha.budgets}_{cfg.hpo.metric}'


def parse_condition_param(condition, ss):
    """
    Parse conditions param to generate ``ConfigSpace.conditions``

    Condition parameters: EqualsCondition, NotEqualsCondition, \
    LessThanCondition, GreaterThanCondition, InCondition

    Args:
        condition (dict): configspace condition dict, which is supposed to
        have four keys for
        ss (CS.ConfigurationSpace): configspace

    Returns:
        ConfigSpace.conditions: the conditions for configspace
    """
    str_func_mapping = {
        'equal': CS.EqualsCondition,
        'not_equal': CS.NotEqualsCondition,
        'less': CS.LessThanCondition,
        'greater': CS.GreaterThanCondition,
        'in': CS.InCondition,
        'and': CS.AndConjunction,
        'or': CS.OrConjunction,
    }
    cond_type = condition['type']
    assert cond_type in str_func_mapping.keys(), f'the param condition ' \
                                                 f'should be in' \
                                                 f' {str_func_mapping.keys()}.'

    if cond_type in ['and', 'in', 'or']:
        return str_func_mapping[cond_type](
            parse_condition_param(condition['child'], ss),
            parse_condition_param(condition['parent'], ss),
        )
    else:
        return str_func_mapping[cond_type](
            child=ss[condition['child']],
            parent=ss[condition['parent']],
            value=condition['value'],
        )


def parse_search_space(config_path):
    """
    Parse yaml format configuration to generate search space

    Arguments:
        config_path (str): the path of the yaml file.
    Return:
        ConfigSpace object: the search space.

    """

    # TODO: add seed for `ConfigurationSpace`
    ss = CS.ConfigurationSpace()
    conditions = []

    with open(config_path, 'r') as ips:
        raw_ss_config = yaml.load(ips, Loader=yaml.FullLoader)

    # Add hyperparameters
    for name in raw_ss_config.keys():
        if name.startswith('condition'):
            # Deal with condition later
            continue
        v = raw_ss_config[name]
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

    # Add conditions
    for name in raw_ss_config.keys():
        if name.startswith('condition'):
            conditions.append(parse_condition_param(raw_ss_config[name], ss))
    ss.add_conditions(conditions)
    return ss


def config2cmdargs(config):
    """
    Arguments:
        config (dict): key is cfg node name, value is the specified value.
    Returns:
        results (list): cmd args
    """

    results = []
    for k, v in config.items():
        results.append(k)
        results.append(v)
    return results


def config2str(config):
    """
    Arguments:
        config (dict): key is cfg node name, value is the choice of
        hyper-parameter.
    Returns:
        name (str): the string representation of this config
    """

    vals = []
    for k in config:
        idx = k.rindex('.')
        vals.append(k[idx + 1:])
        vals.append(str(config[k]))
    name = '_'.join(vals)
    return name


def arm2dict(kvs):
    """
    Arguments:
        kvs (dict): key is hyperparameter name in the form aaa.bb.cccc,
                    and value is the choice.
    Returns:
        config (dict): the same specification for creating a cfg node.
    """

    results = dict()

    for k, v in kvs.items():
        names = k.split('.')
        cur_level = results
        for i in range(len(names) - 1):
            ln = names[i]
            if ln not in cur_level:
                cur_level[ln] = dict()
            cur_level = cur_level[ln]
        cur_level[names[-1]] = v

    return results


def summarize_hpo_results(configs,
                          perfs,
                          white_list=None,
                          desc=False,
                          use_wandb=False):
    if white_list is not None:
        cols = list(white_list) + ['performance']
    else:
        cols = [k for k in configs[0]] + ['performance']

    d = []
    for trial_cfg, result in zip(configs, perfs):
        if white_list is not None:
            d.append([
                trial_cfg[k] if k in trial_cfg.keys() else None
                for k in white_list
            ] + [result])
        else:
            d.append([trial_cfg[k] for k in trial_cfg] + [result])
    d = sorted(d, key=lambda ele: ele[-1], reverse=desc)
    df = pd.DataFrame(d, columns=cols)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    if use_wandb:
        import wandb
        table = wandb.Table(dataframe=df)
        wandb.log({'ConfigurationRank': table})

    return df


def parse_logs(file_list):
    import numpy as np
    import matplotlib.pyplot as plt

    FONTSIZE = 40
    MARKSIZE = 25

    def process(file_path):
        history = []
        with open(file_path, 'r') as F:
            for line in F:
                try:
                    state, line = line.split('INFO: ')
                    config = eval(line[line.find('{'):line.find('}') + 1])
                    performance = float(
                        line[line.find('performance'):].split(' ')[1])
                    print(config, performance)
                    history.append((config, performance))
                except:
                    continue
        best_seen = np.inf
        tol_budget, tmp_b = 0, 0
        x, y = [], []

        for config, performance in history:
            tol_budget += config['federate.total_round_num']
            if best_seen > performance or config[
                    'federate.total_round_num'] > tmp_b:
                best_seen = performance
            x.append(tol_budget)
            y.append(best_seen)
            tmp_b = config['federate.total_round_num']
        return np.array(x) / tol_budget, np.array(y)

    # Draw
    plt.figure(figsize=(10, 7.5))
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.xlabel('Fraction of budget', size=FONTSIZE)
    plt.ylabel('Loss', size=FONTSIZE)

    for file in file_list:
        x, y = process(file)
        plt.plot(x, y, linewidth=1, markersize=MARKSIZE)
    plt.legend(file_list, fontsize=23, loc='lower right')
    plt.savefig('exp2.pdf', bbox_inches='tight')
    plt.close()


def eval_in_fs(cfg, config, budget, config_id, ss, client_cfgs=None):
    """

    Args:
        cfg: fs cfg
        config: sampled trial CS.Configuration
        budget: budget round for this trial
        config_id: Identifier to generate somadditional files
        ss: search space of HPO
        client_cfgs: client-wise cfg

    Returns:
        The best results returned from FedRunner
    """
    import ConfigSpace as CS
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.worker_builder import \
        get_client_cls, get_server_cls
    from federatedscope.core.auxiliaries.runner_builder import get_runner
    from os.path import join as osp

    if isinstance(config, CS.Configuration):
        config = dict(config)
    if 'wrap' in cfg.hpo.scheduler:
        logger.info('scheduler wrapped by FedEx.')
        config['hpo.fedex.ss'] = osp(
            cfg.hpo.working_folder, f"{config_id}_tmp_grid_search_space.yaml")
        if not os.path.exists(config['hpo.fedex.ss']):
            generate_arm(cfg, config, config_id, ss)

        config['federate.save_to'] = osp(cfg.hpo.working_folder,
                                         f"idx_{config_id}.pth")
        config['federate.restore_from'] = osp(cfg.hpo.working_folder,
                                              f"idx_{config_id}.pth")
    if 'hpo.table.idx' in config.keys():
        idx = config['hpo.table.idx']
        config['hpo.fedex.ss'] = osp(cfg.hpo.working_folder,
                                     f"{idx}_tmp_grid_search_space.yaml")
        config['federate.save_to'] = osp(cfg.hpo.working_folder,
                                         f"idx_{idx}.pth")
        config['federate.restore_from'] = osp(cfg.hpo.working_folder,
                                              f"idx_{idx}.pth")
    # Global cfg
    trial_cfg = cfg.clone()
    # specify the configuration of interest
    if cfg.hpo.personalized_ss:
        if isinstance(client_cfgs, CS.Configuration):
            client_cfgs.merge_from_list(config2cmdargs(config))
        else:
            client_cfgs = CfgNode(flatten2nestdict(config))
    else:
        trial_cfg.merge_from_list(config2cmdargs(config))

    # specify the budget
    trial_cfg.merge_from_list(
        ["federate.total_round_num",
         int(budget), "eval.freq",
         int(budget)])
    setup_seed(trial_cfg.seed)
    data, modified_config = get_data(config=trial_cfg.clone())
    trial_cfg.merge_from_other_cfg(modified_config)
    trial_cfg.freeze()
    fed_runner = get_runner(data=data,
                            server_class=get_server_cls(trial_cfg),
                            client_class=get_client_cls(trial_cfg),
                            config=trial_cfg.clone(),
                            client_configs=client_cfgs)
    results = fed_runner.run()

    return results


def generate_arm(cfg, config, config_id, ss):
    def make_local_perturbation(config):
        neighbor = dict()
        for k in config:
            if 'fedex' in k or 'fedopt' in k or k in [
                    'federate.save_to', 'federate.total_round_num', 'eval.freq'
            ]:
                # a workaround
                continue
            hyper = ss.get(k)
            if isinstance(hyper, CS.UniformFloatHyperparameter):
                lb, ub = hyper.lower, hyper.upper
                diameter = cfg.hpo.fedex.wrapper.eps * (ub - lb)
                new_val = (config[k] -
                           0.5 * diameter) + np.random.uniform() * diameter
                neighbor[k] = float(np.clip(new_val, lb, ub))
            elif isinstance(hyper, CS.UniformIntegerHyperparameter):
                lb, ub = hyper.lower, hyper.upper
                diameter = cfg.hpo.fedex.wrapper.eps * (ub - lb)
                new_val = round(
                    float((config[k] - 0.5 * diameter) +
                          np.random.uniform() * diameter))
                neighbor[k] = int(np.clip(new_val, lb, ub))
            elif isinstance(hyper, CS.CategoricalHyperparameter):
                if len(hyper.choices) == 1:
                    neighbor[k] = config[k]
                else:
                    threshold = cfg.hpo.fedex.wrapper.eps * len(
                        hyper.choices) / (len(hyper.choices) - 1)
                    rn = np.random.uniform()
                    new_val = np.random.choice(
                        hyper.choices) if rn <= threshold else config[k]
                    if type(new_val) in [np.int32, np.int64]:
                        neighbor[k] = int(new_val)
                    elif type(new_val) in [np.float32, np.float64]:
                        neighbor[k] = float(new_val)
                    else:
                        neighbor[k] = str(new_val)
            else:
                raise TypeError("Value of {} has an invalid type {}".format(
                    k, type(config[k])))

        return neighbor

    arms = dict(("arm{}".format(1 + j), make_local_perturbation(config))
                for j in range(cfg.hpo.fedex.wrapper.arm - 1))
    arms['arm0'] = dict((k, v) for k, v in config.items() if k in arms['arm1'])
    with open(
            os.path.join(cfg.hpo.working_folder,
                         f'{config_id}_tmp_grid_search_space.yaml'), 'w') as f:
        yaml.dump(arms, f)


def flatten2nestdict(raw_dict, delimiter='.'):
    # TODO: delete this for the function of `arm2dict`
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    new_dict = dict()
    for key, value in raw_dict.items():
        keys = key.split(delimiter)
        nested_set(new_dict, keys, value)
    return new_dict


def config_bool2int(config):
    # TODO: refactor bool/str to int
    import copy
    new_dict = copy.deepcopy(config)
    for key, value in new_dict.items():
        if isinstance(new_dict[key], bool):
            new_dict[key] = int(value)
    return new_dict


def log2wandb(trial, config, results, trial_cfg):
    import wandb
    key1, key2 = trial_cfg.hpo.metric.split('.')
    log_res = {
        'Trial_index': trial,
        'Config': config_bool2int(config),
        trial_cfg.hpo.metric: results[key1][key2],
    }
    wandb.log(log_res)
