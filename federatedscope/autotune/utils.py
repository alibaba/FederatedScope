import yaml
import pandas as pd
import ConfigSpace as CS


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
        config (dict): key is cfg node name, value is the choice of
        hyper-parameter.
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
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    return df


def parse_logs(file_list):
    import numpy as np
    import matplotlib.pyplot as plt

    FONTSIZE = 40
    MARKSIZE = 25

    def process(file):
        history = []
        with open(file, 'r') as F:
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


def eval_in_fs(cfg, config, budget):
    import ConfigSpace as CS
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.worker_builder import \
        get_client_cls, get_server_cls
    from federatedscope.core.fed_runner import get_runner
    from federatedscope.autotune.utils import config2cmdargs
    from os.path import join as osp

    if isinstance(config, CS.Configuration):
        config = dict(config)
    # Add FedEx related keys to config
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
    Fed_runner = get_runner(data=data,
                            server_class=get_server_cls(trial_cfg),
                            client_class=get_client_cls(trial_cfg),
                            config=trial_cfg.clone())
    results = Fed_runner.run()
    key1, key2 = trial_cfg.hpo.metric.split('.')
    return results[key1][key2]
