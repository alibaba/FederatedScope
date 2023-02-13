import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

FONTSIZE = 40
MARKSIZE = 25

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

TRIAL2SEED = {
    't0': 12345,
    't1': 12346,
    't2': 12347,
}

COLORS = [
    u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#8c564b', u'#9467bd'
]


def parse_logs(root,
               file_list,
               eval_key='client_summarized_weighted_avg',
               suf=''):
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


def args_run_from_scratch(root,
                          file_list,
                          fs_yaml='********',
                          suf='',
                          device=8):
    def get_args(root, logs):
        log_file = os.path.join(root, logs)
        arg_key = []
        arg_val = []
        with open(log_file, 'r') as f:
            flag = False
            cnt = 0
            for line in f.readlines():
                if 'wrap' in log_file:
                    if 'Winner config_id:' in line:
                        idx = line.split('Winner config_id: ')[-1]
                        if idx.endswith(' ') or idx.endswith('\n'):
                            idx = idx[:-1]
                        break
                else:
                    if '== HPO Final ==' in line:
                        flag = True

                    if flag:
                        if line.startswith(' '):
                            args = [x for x in line.split(' ') if len(x)]
                            arg_key += args
                        if line.startswith('0 '):
                            args = [x for x in line.split(' ') if len(x)][1:]
                            arg_val += args

        if 'wrap' in log_file:
            fedex_yaml = os.path.join(root, f'idx_{idx}_fedex.yaml')
            arm_yaml = os.path.join(root, f'{idx}_tmp_grid_search_space.yaml')

            with open(fedex_yaml, 'r') as y:
                fedex = dict(yaml.load(y, Loader=yaml.FullLoader))

            with open(arm_yaml, 'r') as y:
                arm = dict(yaml.load(y, Loader=yaml.FullLoader))

            best_arm = np.argmax(fedex['z'][0])
            arg = ''
            for key, val in arm[f'arm{best_arm}'].items():
                arg += f'{key} {val} '
        else:
            arg_key = [x for x in arg_key if not x.endswith('\n')]
            arg_val = [x for x in arg_val if not x.endswith('\n')]
            arg = ''
            for key, val in zip(arg_key[:-1], arg_val[:-1]):
                arg += f'{key} {val} '
        return arg

    d = 0
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
            args = get_args(
                os.path.join(prefix, f'{file}{suf}'),
                os.path.join(
                    f'{new_name}_{budget}_server_global_eval.val_avg_loss',
                    'exp_print.log'))

            seed = TRIAL2SEED[prefix]
            print(
                f'nohup python federatedscope/main.py --cfg {fs_yaml} {args} device {d % device} > {file}_{seed}{suf}.log &'
            )
            d += 1


if __name__ == '__main__':
    parse_logs(list(TRIAL2SEED.keys()),
               list(METHOD.keys()),
               eval_key='server_global_eval',
               suf='_pubmed_avg')

    args_run_from_scratch(
        list(TRIAL2SEED.keys()),
        list(METHOD.keys()),
        fs_yaml=
        'scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml',
        suf='_pubmed_avg')
