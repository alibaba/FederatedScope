import copy
import time
import yaml
import logging
import numpy as np
import pandas as pd
import ConfigSpace as CS

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

    if cond_type in ['and', 'in']:
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


def eval_in_fs(cfg, config, budget, client_cfgs=None, trial_index=0):
    """

    Args:
        cfg: fs cfg
        config: sampled trial CS.Configuration
        budget: budget round for this trial
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
    # Add FedEx related keys to config
    if 'hpo.table.idx' in config.keys():
        idx = config['hpo.table.idx']
        config['hpo.fedex.ss'] = osp(cfg.hpo.working_folder,
                                     f"{idx}_tmp_grid_search_space.yaml")
        config['federate.save_to'] = osp(cfg.hpo.working_folder,
                                         f"idx_{idx}.pth")
        config['federate.restore_from'] = osp(cfg.hpo.working_folder,
                                              f"idx_{idx}.pth")
    config['hpo.trial_index'] = trial_index

    # Global cfg
    trial_cfg = cfg.clone()
    # specify the configuration of interest
    trial_cfg.merge_from_list(config2cmdargs(config))
    # specify the budget
    trial_cfg.merge_from_list(["federate.total_round_num", int(budget)])
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


def config_bool2int(config):
    import copy
    new_dict = copy.deepcopy(config)
    for key, value in new_dict.items():
        if isinstance(new_dict[key], bool):
            new_dict[key] = int(value)
    return new_dict


def adjust_lightness(color, num=0.5):
    import colorsys
    import matplotlib.colors as mc

    MAX_NUM = 1.8
    num = min(num, MAX_NUM)
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, num * c[1])), c[2])


def log2wandb(trial, config, results, trial_cfg, df):
    import wandb
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    FONTSIZE = 30
    MARKSIZE = 200

    # Base information
    key1, key2 = trial_cfg.hpo.metric.split('.')
    log_res = {
        'Trial_index': trial,
        'Config': config_bool2int(config),
        trial_cfg.hpo.metric: results[key1][key2],
    }

    # Diagnosis with 1d landscape
    landscape_1d = {}
    if trial_cfg.hpo.diagnosis.use:
        col_name = df.columns
        num_results = df.shape[0]
        step = num_results

        # Return when number of results are too less
        if step < 0:
            return

        # 1D landscape
        for hyperparam in trial_cfg.hpo.diagnosis.landscape_1d:
            if hyperparam not in col_name:
                logger.warning(f'Invalid hyperparam name: {hyperparam}')
                continue
            else:
                plt.figure(figsize=(20, 15))
                ranks = list(
                    df.groupby(hyperparam)["performance"].mean().fillna(
                        0).sort_values()[::-1].index)
                ranks.reverse()
                sns.boxplot(x="performance",
                            y=hyperparam,
                            data=df,
                            order=ranks,
                            width=.2,
                            saturation=0.5)
                sns.stripplot(x="performance",
                              y=hyperparam,
                              data=df,
                              jitter=True,
                              color="black",
                              size=10,
                              linewidth=0,
                              order=ranks)
                plt.yticks(rotation=45, fontsize=FONTSIZE)
                plt.xticks(fontsize=FONTSIZE)
                plt.xlabel(trial_cfg.hpo.metric, size=FONTSIZE)
                plt.ylabel("", size=FONTSIZE)
                plt.title(f"{hyperparam} - Rank", fontsize=FONTSIZE)
                sns.despine(trim=True)
                landscape_1d[f"{hyperparam}"] = wandb.Image(plt.gcf())
                plt.close()

    # PCA of exploration
    if trial_cfg.hpo.diagnosis.use:
        from sklearn import preprocessing
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

        X = df.iloc[:, :-1]

        for col in X.columns.tolist():
            X[col] = X[col].astype('category')
            X[col] = X[col].cat.codes
        X_std = preprocessing.scale(X)
        pca = PCA(n_components=1)
        pca.fit(X_std)
        X_pca = pd.DataFrame(
            pca.fit_transform(X_std)).rename(columns={0: 'component'})
        Y = pd.DataFrame(df["performance"])
        data_pca = pd.concat([X_pca, Y], axis=1)

        kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
        reg = GaussianProcessRegressor(kernel=kernel,
                                       n_restarts_optimizer=10,
                                       alpha=0.1)
        reg.fit([[x] for x in data_pca['component'].tolist()],
                data_pca['performance'].tolist())
        x_ticks = np.linspace(np.min(data_pca['component']),
                              np.max(data_pca['component']), 100)
        ys = reg.predict([[x] for x in x_ticks])

        plt.figure(figsize=(20, 15))
        sns.scatterplot(data=data_pca,
                        x='component',
                        y='performance',
                        s=MARKSIZE)
        gp = pd.DataFrame(dict(x=x_ticks, y=ys))
        sns.lineplot(data=gp, x='x', y='y')

        plt.title("Gaussian", fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.xlabel("ConfigSpace", size=FONTSIZE)
        plt.ylabel("Loss", size=FONTSIZE)
        pca = wandb.Image(plt.gcf())
        plt.close()

    # Text info guidance
    if trial_cfg.hpo.diagnosis.use:
        plt.figure(figsize=(30, 15))
        anc_x, anc_y, bias = 0.5, 0.95, 0
        texts = [
            "Searching the optimal federated configuration automatically...",
            f"Trial [{trial}] ongoing", "The configurations being used are:",
            config,
            "For detailed information, please see Diagnosis and Autotune."
        ]

        for t in texts:
            if isinstance(t, str):
                plt.text(anc_x,
                         anc_y + bias,
                         t,
                         size=50,
                         ha="center",
                         va="center",
                         bbox=dict(
                             boxstyle="sawtooth",
                             facecolor='lightblue',
                             edgecolor='black',
                         ))
                bias -= 0.1
            elif isinstance(t, dict):
                for key, value in t.items():
                    plt.text(anc_x,
                             anc_y + bias,
                             f"{key}: {value}",
                             size=50,
                             ha='center',
                             va="center",
                             bbox=dict(
                                 boxstyle="sawtooth",
                                 facecolor='none',
                                 edgecolor='black',
                             ))
                    bias -= 0.1

        plt.axis('off')
        info = wandb.Image(plt.gcf())
        plt.close()

    # Parallel coordinates
    new_df = copy.deepcopy(df)
    px_layout = []
    for col in new_df.columns.tolist():
        if isinstance(new_df[col][0], str):
            new_df[col] = new_df[col].astype('category')
            cat_map = dict(zip(new_df[col].cat.codes, new_df[col]))
            px_layout.append({
                'range': [min(cat_map.keys()),
                          max(cat_map.keys())],
                'label': col,
                'tickvals': list(cat_map.keys()),
                'ticktext': list(cat_map.values()),
                'values': new_df[col].cat.codes,
            })
        else:
            px_layout.append({
                'range': [0, max(new_df[col])],
                'label': col,
                'values': new_df[col],
            })
    new_df['Trial Index'] = range(1, len(new_df) + 1)

    px_fig = go.Figure(data=go.Parcoords(line=dict(color=new_df['Trial Index'],
                                                   colorscale='Electric',
                                                   showscale=True,
                                                   cmin=1,
                                                   cmax=len(new_df) + 1),
                                         dimensions=px_layout))
    plotly_html = 'test.html'
    px_fig.write_html(plotly_html, auto_play=False)
    para_coo = wandb.Html(open(plotly_html))

    # Loss, lower the better
    best_perf = np.min(df['performance'])

    wandb.log({
        'pca': pca,
        'info': info,
        'delimiter': 1.0,
        'best_perf': best_perf,
        'para_coo': para_coo,
        **log_res,
        **landscape_1d
    })

    time.sleep(3)
