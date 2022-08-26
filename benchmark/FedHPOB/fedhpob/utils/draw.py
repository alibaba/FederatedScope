import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

FONTSIZE = 30
MARKSIZE = 25
COLORS = [
    u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', 'black',
    u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'
]


def logloader(file):
    log = []
    with open(file) as f:
        file = f.readlines()
        for line in file:
            line = json.loads(s=line)
            log.append(line)
    return log


def ecdf(model, data_list, algo, sample_client=None, key='test_acc'):
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
            try:
                best_round = np.argmin(val_loss)
            except:
                continue
            target.append(result[key][best_round])
        norm_regret = np.sort(1 - (np.array(target) / np.max(target)))
        y = np.arange(len(norm_regret)) / float(len(norm_regret) - 1)
        plt.plot(norm_regret, y)
    if data_list[0].endswith('datasets'):
        legend = ['CoLA', 'SST2']
    elif data_list[0].endswith('openml'):
        legend = [x.split('@')[0] for x in data_list]
    elif data_list[0].lower == 'cora':
        legend = ['Cora', 'CiteSeer', 'PubMed']
    else:
        legend = [x.upper() for x in data_list]
    # Not show legend
    del legend
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{model}_{sample_client}_{algo}_cdf.pdf',
                bbox_inches='tight')
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
            rank_bbo = rankdata(ys[:6], axis=0)
            rank_mf = rankdata(ys[-5:], axis=0)
        else:
            rank += rankdata(ys, axis=0)
            rank_bbo += rankdata(ys[:6], axis=0)
            rank_mf += rankdata(ys[-5:], axis=0)
    xs = np.linspace(0, 1, 500)
    return xs, rank / len(traj_dict), rank_bbo / len(traj_dict), rank_mf / len(
        traj_dict)


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
            rank_bbo = np.array(ys[:6])
            rank_mf = np.array(ys[-5:])
        else:
            rank += np.array(ys)
            rank_bbo += np.array(ys[:6])
            rank_mf += np.array(ys[-5:])
    xs = np.linspace(0, 1, 500)
    print(
        str([round(x, 4) for x in (rank / len(traj_dict))[:, -1].tolist()
             ]).replace(',', ' &'))
    return xs, rank / len(traj_dict), rank_bbo / len(traj_dict), rank_mf / len(
        traj_dict)


def draw_rank(mean_ranks, mean_ranks_bbo, mean_ranks_mf, xs, opt_all, dataset,
              family, suffix, Y_label):
    os.makedirs('figures', exist_ok=True)
    # BBO + MF
    plt.figure(figsize=(10, 7.5))

    for idx, rank in enumerate(mean_ranks):
        plt.plot(xs, rank, linewidth=1, color=COLORS[idx], markersize=MARKSIZE)
    plt.xticks(np.linspace(0, 1, 5),
               labels=['1e-4', '1e-3', '1e-2', '1e-1', '1'],
               fontsize=FONTSIZE)
    if Y_label == 'Mean_rank':
        plt.yticks(np.linspace(1, 10, 10),
                   ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                   fontsize=FONTSIZE)
        plt.ylabel('Mean rank', size=FONTSIZE)
        print(dataset, mean_ranks[:, -1].tolist())
    else:
        plt.yticks(fontsize=FONTSIZE)

    plt.xlabel('Fraction of budget', size=FONTSIZE)
    plt.savefig(
        f"figures/{dataset.replace('@', '_')}_{family}_"
        f"{suffix}_over_time_all_{Y_label}.pdf",
        bbox_inches='tight')
    plt.close()

    # BBO
    plt.figure(figsize=(10, 7.5))

    for idx, rank in enumerate(mean_ranks_bbo):
        plt.plot(xs, rank, linewidth=1, color=COLORS[idx], markersize=MARKSIZE)
    plt.xticks(np.linspace(0, 1, 5),
               labels=['1e-4', '1e-3', '1e-2', '1e-1', '1'],
               fontsize=FONTSIZE)
    if Y_label == 'Mean_rank':
        plt.yticks(np.linspace(1, 5, 5), ['1', '2', '3', '4', '5'],
                   fontsize=FONTSIZE)
        plt.ylabel('Mean rank', size=FONTSIZE)
    else:
        plt.yticks(fontsize=FONTSIZE)
        plt.ylabel(Y_label, size=FONTSIZE)

    plt.xlabel('Fraction of budget', size=FONTSIZE)
    plt.savefig(
        f"figures/{dataset.replace('@', '_')}_{family}_"
        f"{suffix}_over_time_bbo_{Y_label}.pdf",
        bbox_inches='tight')
    plt.close()

    # MF
    plt.figure(figsize=(10, 7.5))

    for idx, rank in enumerate(mean_ranks_mf):
        plt.plot(xs,
                 rank,
                 linewidth=1,
                 color=COLORS[idx + 6],
                 markersize=MARKSIZE)
    plt.xticks(np.linspace(0, 1, 5),
               labels=['1e-4', '1e-3', '1e-2', '1e-1', '1'],
               fontsize=FONTSIZE)
    if Y_label == 'Mean_rank':
        plt.yticks(np.linspace(1, 5, 5), ['1', '2', '3', '4', '5'],
                   fontsize=FONTSIZE)
        plt.ylabel('Mean rank', size=FONTSIZE)
    else:
        plt.yticks(fontsize=FONTSIZE)
        plt.ylabel(Y_label, size=FONTSIZE)

    plt.xlabel('Fraction of budget', size=FONTSIZE)
    plt.savefig(
        f"figures/{dataset.replace('@', '_')}_{family}_"
        f"{suffix}_over_time_mf_{Y_label}.pdf",
        bbox_inches='tight')
    plt.close()


def rank_over_time(root,
                   family='all',
                   mode='tabular',
                   algo='avg',
                   repeat=5,
                   loss=False):
    suffix = f'{mode}_{algo}'
    if family == 'cnn':
        data_list = ['femnist']
    elif family == 'gcn':
        data_list = ['cora', 'citeseer', 'pubmed']
    elif family == 'bert':
        data_list = ['sst2@huggingface_datasets', 'cola@huggingface_datasets']
    elif family in ['mlp', 'lr']:
        data_list = [
            '10101@openml', '53@openml', '146818@openml', '146821@openml',
            '146822@openml', '31@openml', '3917@openml'
        ]
    else:
        # All dataset
        data_list = [
            'femnist', 'cora', 'citeseer', 'pubmed', 'sst2', 'cola', 'mlp',
            'lr', '10101@openml', '53@openml', '146818@openml',
            '146821@openml', '9952@openml', '146822@openml', '31@openml',
            '3917@openml'
        ]

    # Please place these logs to one dir
    bbo = ['RS', 'BO_GP', 'BO_RF', 'BO_KDE', 'DE', 'grid_search']
    mf = ['HB', 'BOHB', 'DEHB', 'TPE_MD', 'TPE_HB']
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
                    if file.startswith(f'{opt.lower()}_'):
                        traj[dataset][i].append(
                            logloader(os.path.join(path, file)))

    # Draw over dataset
    family_rank = []
    family_rank_bbo = []
    family_rank_mf = []
    for dataset in traj:
        if loss:
            print(dataset)
            xs, mean_ranks, mean_ranks_bbo, mean_ranks_mf = get_mean_loss(
                traj[dataset])
            Y_label = 'Loss'
        else:
            xs, mean_ranks, mean_ranks_bbo, mean_ranks_mf = get_mean_rank(
                traj[dataset])
            Y_label = 'Mean_rank'
        if len(family_rank):
            family_rank += mean_ranks
            family_rank_bbo += mean_ranks_bbo
            family_rank_mf += mean_ranks_mf
        else:
            family_rank, family_rank_bbo, family_rank_mf = mean_ranks, \
                                                           mean_ranks_bbo, \
                                                           mean_ranks_mf
        draw_rank(mean_ranks, mean_ranks_bbo, mean_ranks_mf, xs, opt_all,
                  dataset, family, suffix, Y_label)

    # Draw entire family
    draw_rank(family_rank / len(traj), family_rank_bbo / len(traj),
              family_rank_mf / len(traj), xs, opt_all, 'entire', family,
              suffix, Y_label)


def landscape(model='cnn',
              dname='femnist',
              algo='avg',
              sample_client=None,
              key='test_acc'):
    import plotly.graph_objects as go
    from fedhpob.config import fhb_cfg
    from fedhpob.benchmarks import TabularBenchmark

    z = []
    benchmark = TabularBenchmark(model, dname, algo, device=-1)

    def get_best_config(benchmark):
        results, config = [], []
        for idx in tqdm(range(len(benchmark.table))):
            row = benchmark.table.iloc[idx]
            if sample_client is not None and row[
                    'sample_client'] != sample_client:
                continue
            result = eval(row['result'])
            val_loss = result['val_avg_loss']
            try:
                best_round = np.argmin(val_loss)
            except:
                continue
            results.append(result[key][best_round])
            config.append(row)
        best_index = np.argmax(results)
        return config[best_index], results[best_index]

    # config, _ = get_best_config(benchmark)
    config = {'wd': 0.0, 'dropout': 0.5, 'step': 1.0}
    config_space = benchmark.get_configuration_space()
    X, Y = sorted(list(config_space['batch'])), sorted(list(
        config_space['lr']))
    print(X, Y)
    for lr in Y:
        y = []
        for batch in X:
            xy = {'lr': lr, 'batch': batch}
            print({**config, **xy})
            res = benchmark({
                **config,
                **xy
            }, {
                'sample_client': 1.0,
                'round': 249
            },
                            fhb_cfg=fhb_cfg,
                            seed=12345)
            y.append(res['function_value'])
        z.append(y)
    Z = np.array(z)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='FEMNIST (FedAvg)',
                      autosize=False,
                      width=900,
                      height=900,
                      margin=dict(l=65, r=50, b=65, t=90),
                      scene=dict(
                          xaxis_title='BS',
                          yaxis_title='LR',
                          zaxis_title='ACC',
                      ))
    fig.write_image(os.path.join('figures', 'femnist_fedavg_landscape.pdf'))

    return


if __name__ == '__main__':
    ecdf('gcn', ['cora', 'citeseer', 'pubmed'], sample_client=1.0)
