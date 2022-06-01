import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

FONTSIZE = 30
MARKSIZE = 25


def logloader(file):
    log = []
    with open(file) as f:
        file = f.readlines()
        for line in file:
            line = json.loads(s=line)
            log.append(line)
    return log


def ecdf(model, data_list, algo, sample_client=None, key='test_acc'):
    import datetime
    from fedhpob.benchmarks import TabularBenchmark

    # Draw ECDF from target data_list
    plt.figure(figsize=(10, 7.5))
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.xlabel('Normalized regret', size=FONTSIZE)
    plt.ylabel('P(X <= x)', size=FONTSIZE)

    # Get target data (tabular only)
    for data in data_list:
        benchmark = TabularBenchmark(model, data, algo, device=-1)
        target = [0]  # Init with zero
        for idx in tqdm(range(len(benchmark.table))):
            row = benchmark.table.iloc[idx]
            if sample_client is not None and row[
                    'sample_client'] != sample_client:
                continue
            result = eval(row['result'])
            val_loss = result['val_avg_loss']
            best_round = np.argmin(val_loss)
            target.append(result[key][best_round])
        norm_regret = np.sort(1 - (np.array(target) / np.max(target)))
        y = np.arange(len(norm_regret)) / float(len(norm_regret) - 1)
        plt.plot(norm_regret, y)
    plt.legend(data_list, fontsize=23, loc='lower right')
    plt.savefig(f'{model}_{sample_client}_cdf.pdf', bbox_inches='tight')
    plt.close()

    return target


def get_mean_rank(traj_dict):
    from scipy.stats import rankdata

    xs = np.logspace(-4, 0, 500)
    # Convert loss to rank
    for repeat in traj_dict:
        logs = traj_dict[repeat]
        ys = []
        for data in logs:
            tol_time = data[-1]['Consumed']
            frac_budget = np.array([i['Consumed'] / tol_time for i in data])
            loss = np.array([i['best_value'] for i in data])
            ys.append([
                loss[np.where(x >= frac_budget)[0][-1]]
                if len(np.where(x >= frac_budget)[0]) != 0 else np.inf
                for x in xs
            ])
        if repeat == 0:
            rank = rankdata(ys, axis=0)
        else:
            rank += rankdata(ys, axis=0)
    xs = np.linspace(0, 1, 500)
    return xs, rank / len(traj_dict)


def get_mean_loss(traj_dict):
    xs = np.logspace(-4, 0, 500)
    for repeat in traj_dict:
        logs = traj_dict[repeat]
        ys = []
        for data in logs:
            tol_time = data[-1]['Consumed']
            frac_budget = np.array([i['Consumed'] / tol_time for i in data])
            loss = np.array([i['best_value'] for i in data])
            ys.append([
                loss[np.where(x >= frac_budget)[0][-1]]
                if len(np.where(x >= frac_budget)[0]) != 0 else np.inf
                for x in xs
            ])
        if repeat == 0:
            rank = np.array(ys)
        else:
            rank += np.array(ys)
    xs = np.linspace(0, 1, 500)
    return xs, rank / len(traj_dict)


def rank_over_time(root,
                   family='all',
                   mode='tabular',
                   algo='avg',
                   repeat=5,
                   loss=False):
    suffix = f'{mode}_{algo}'
    if family == 'cnn':
        data_list = ['femnist', 'cifar10']
    elif family == 'gcn':
        data_list = ['cora', 'citeseer', 'pubmed']
    elif family == 'transformer':
        data_list = ['sst2', 'cola']
    elif family in ['mlp', 'lr']:
        data_list = [
            '10101@openml', '53@openml', '146818@openml', '146821@openml',
            '9952@openml', '146822@openml', '31@openml', '3917@openml'
        ]
    else:
        # All dataset
        data_list = [
            'femnist', 'cifar10', 'cora', 'citeseer', 'pubmed', 'sst2', 'cola',
            'mlp', 'lr', '10101@openml', '53@openml', '146818@openml',
            '146821@openml', '9952@openml', '146822@openml', '31@openml',
            '3917@openml'
        ]

    # Please place these logs to one dir
    bbo = ['rs', 'bo_gp', 'bo_rf', 'bo_kde', 'de']
    mf = ['hb', 'bohb', 'dehb', 'tpe_md', 'tpe_hb']
    opt_all = bbo + mf

    # Load trajectories
    traj = {}
    for dataset in data_list:
        traj[dataset] = {}
        path = os.path.join(root, f'{dataset}_{family}_{suffix}')
        files = os.listdir(path)
        files = [file for file in files if file.endswith('.txt')]
        for i in range(repeat):
            files_i = [file for file in files if f'repeat{i}' in file]
            traj[dataset][i] = []
            for opt in opt_all:
                for file in files_i:
                    if file.startswith(opt):
                        traj[dataset][i].append(
                            logloader(os.path.join(path, file)))

    # Draw over dataset
    for dataset in traj:
        if loss:
            xs, mean_ranks = get_mean_loss(traj[dataset])
        else:
            xs, mean_ranks = get_mean_rank(traj[dataset])
        plt.figure(figsize=(10, 7.5))
        plt.xticks(np.linspace(0, 1, 5),
                   labels=['1e-4', '1e-3', '1e-2', '1e-1', '1'],
                   fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)

        plt.xlabel('Fraction of budget', size=FONTSIZE)
        plt.ylabel('Mean rank', size=FONTSIZE)

        for rank in mean_ranks:
            plt.plot(xs, rank, linewidth=1, markersize=MARKSIZE)
        plt.legend(opt_all, fontsize=23, loc='lower right', bbox_to_anchor=(1.35, 0))
        plt.savefig(f'{dataset}_{family}_{suffix}_rank_over_time.pdf',
                    bbox_inches='tight')
        # plt.show()
        plt.close()


if __name__ == '__main__':
    ecdf('gcn', ['cora', 'citeseer', 'pubmed'], sample_client=1.0)