import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

FONTSIZE = 40
MARKSIZE = 25


def logloader(file):
    log = []
    with open(file) as f:
        file = f.readlines()
        for line in file:
            line = json.loads(s=line)
            log.append(line)
    return log


def ecdf(model, data_list, sample_client=None, key='test_acc'):
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
        benchmark = TabularBenchmark(model, data, device=-1)
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


def rank_over_time(root):
    # Please place these logs to one dir
    target_opt = [
        'rs', 'bo_gp', 'bo_rf', 'bo_kde', 'de', 'hb', 'bohb', 'dehb', 'tpe_md',
        'tpe_hb'
    ]
    files = os.listdir(root)
    logs = []
    for opt in target_opt:
        for file in files:
            if file.startswith(opt):
                logs.append(logloader(file))
                break

    # Draw over time
    plt.figure(figsize=(10, 7.5))
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.xlabel('Fraction of budget', size=FONTSIZE)
    plt.ylabel('Mean rank', size=FONTSIZE)

    for data in logs:
        tol_time = data[-1]['Consumed']
        frac_budget = np.array([i['Consumed'] / tol_time for i in data])
        # TODO: sort by rank
        loss = np.array([i['best_value'] for i in data])
        plt.plot(frac_budget, loss, linewidth=1, markersize=MARKSIZE)
    plt.legend(target_opt, fontsize=23, loc='lower right')
    plt.savefig(f'{root}_rank_over_time.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    ecdf('gcn', ['cora', 'citeseer', 'pubmed'], sample_client=None)
