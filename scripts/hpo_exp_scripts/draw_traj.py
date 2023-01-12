METHOD = {
    'rs': 'RS',
    'bo_gp': 'BO_GP',
    'bo_rf': 'BO_RF',
    'bo_kde': 'BO_KDE',
    'hb': 'HB',
    'bohb': 'BOHB',
    'rs_wrap': 'RS+FedEx',
    'bo_gp_wrap': 'BO_GP+FedEx',
    'bo_rf_wrap': 'BO_RF+FedEx',
    'bo_kde_wrap': 'BO_KDE+FedEx',
    'hb_wrap': 'HB+FedEx',
    'bohb_wrap': 'BOHB+FedEx',
}

COLORS = [
    u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#8c564b', u'#9467bd'
]


def parse_logs(root,
               file_list,
               eval_key='client_summarized_weighted_avg',
               suf=''):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    FONTSIZE = 40
    MARKSIZE = 25

    def process(file):
        history = []
        with open(file, 'r') as F:
            F = F.readlines()
            for idx, line in tqdm(enumerate(F)):
                if "'Round': 'Final'" in line:
                    last_line = F[idx - 2]
                    _, last_line = last_line.split('INFO: ')
                    last_line = eval(last_line)
                    config = {'federate.total_round_num': last_line['Round']}
                    try:
                        state, line = line.split('INFO: ')
                        results = eval(line)
                        performance = results['Results_raw'][eval_key][
                            'val_avg_loss']
                        history.append((config, performance))
                    except Exception as error:
                        continue
        best_seen = np.inf
        tol_budget = 0
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

    data_x = {}
    data_y = {}
    for file in file_list:
        data_x[file] = []
        data_y[file] = []

    for prefix in root:
        for file in file_list:
            if 'wrap' in file:
                new_name = f'wrap_{file[:-5]}'
            else:
                new_name = file

            if 'hb' in file:
                budget = [9, 81]
            else:
                budget = [50, 50]
            x, y = process(
                os.path.join(
                    prefix, f'{file}{suf}',
                    f'{new_name}_{budget}_server_global_eval.val_avg_loss',
                    'exp_print.log'))
            data_x[file].append(x)
            data_y[file].append(y)

    for i, file in enumerate(file_list):
        if 'wrap' in file:
            linestyle = '--'
        else:
            linestyle = '-'
        plt.plot(np.mean(data_x[file], axis=0),
                 np.mean(data_y[file], axis=0),
                 linewidth=1,
                 color=COLORS[i % len(COLORS)],
                 markersize=MARKSIZE,
                 linestyle=linestyle)
    if file_list[0].endswith('_opt.log'):
        suffix = 'opt'
    else:
        suffix = 'avg'
    plt.xscale("log")
    plt.xticks([0.01, 0.1, 1], ['1e-2', '1e-1', '1'])
    plt.legend(list(METHOD.values()),
               loc='upper right',
               prop={'size': 22},
               bbox_to_anchor=(1.5, 1),
               borderaxespad=0)
    plt.savefig(f'exp2_{suffix}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


parse_logs(['t0', 't1', 't2'],
           list(METHOD.keys()),
           eval_key='server_global_eval',
           suf='_pubmed_avg')
