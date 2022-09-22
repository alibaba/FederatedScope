import os
import gzip
import pickle
import re
from datetime import *

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(root, model, dname, algo):
    path = os.path.join(root, model, dname, algo)
    datafile = os.path.join(path, 'tabular.csv.gz')
    infofile = os.path.join(path, 'info.pkl')

    if not os.path.exists(datafile):
        df = logs2df(dname, path)
        df.to_csv(datafile, index=False, compression='gzip')
    if not os.path.exists(infofile):
        info = logs2info(dname, path)
        pkl = pickle.dumps(info)
        with open(infofile, 'wb') as f:
            f.write(pkl)

    df = pd.read_csv(datafile)
    with open(infofile, 'rb') as f:
        info = pickle.loads(f.read())

    return df, info


def logs2info(dname, root, sample_client_rate=[0.2, 0.4, 0.6, 0.8, 1.0]):
    sample_client_rate = set(sample_client_rate)
    dir_names = [f'out_{dname}_' + str(x) for x in sample_client_rate]
    trail_names = [x for x in os.listdir(os.path.join(root, dir_names[0]))]
    split_names = [x.split('_') for x in trail_names if x.startswith('lr')]
    args = [''.join(re.findall(r'[A-Za-z]', arg)) for arg in split_names[0]]

    search_space = {
        arg: set([float(x[i][len(arg):]) for x in split_names])
        for i, arg in enumerate(args)
    }
    if dname in ['cola', 'sst2']:
        fidelity_space = {
            'sample_client': set(sample_client_rate),
            'round': [x for x in range(40)]
        }
        eval_freq = 1
    elif dname in ['femnist']:
        fidelity_space = {
            'sample_client': set(sample_client_rate),
            'round': [x + 1 for x in range(0, 500, 2)]
        }
        eval_freq = 2
    else:
        fidelity_space = {
            'sample_client': set(sample_client_rate),
            'round': [x for x in range(500)]
        }
        eval_freq = 1
    info = {
        'configuration_space': search_space,
        'fidelity_space': fidelity_space,
        'eval_freq': eval_freq
    }

    return info


def read_fairness(lines):
    fairness_list = []
    for line in lines:
        tmp_line = str(line)
        if 'Server' in tmp_line:
            results = eval(line)
            new_results = {}
            for key in results['Results_raw']:
                new_results[f'{key}_fair'] = results['Results_raw'][key]
            fairness_list.append(new_results)
    return fairness_list


def logs2df(dname,
            root='',
            sample_client_rate=[0.2, 0.4, 0.6, 0.8, 1.0],
            metrics=[
                'train_avg_loss', 'val_avg_loss', 'test_avg_loss', 'train_acc',
                'val_acc', 'test_acc', 'train_f1', 'val_f1', 'test_f1',
                'fairness'
            ]):
    sample_client_rate = [str(round(x, 1)) for x in sample_client_rate]
    dir_names = [f'out_{dname}_' + str(x) for x in sample_client_rate]

    trail_names = [x for x in os.listdir(os.path.join(root, dir_names[0]))]
    split_names = [x.split('_') for x in trail_names if x.startswith('lr')]

    args = [''.join(re.findall(r'[A-Za-z]', arg)) for arg in split_names[0]]
    df = pd.DataFrame(None, columns=['sample_client'] + args + ['result'])

    print('Processing...')
    cnt = 0
    for name, rate in zip(dir_names, sample_client_rate):
        path = os.path.join(root, name)
        trail_names = sorted(
            [x for x in os.listdir(path) if x.startswith('lr')])
        # trail_names = group_by_seed(trail_names)
        for file_name in tqdm(trail_names):
            metrics_dict = {x: [] for x in metrics}
            time_dict = {
                x: []
                for x in ['train_time', 'eval_time', 'tol_time']
            }
            # Load fairness-related metric if exists.
            fairness_list = None
            fairness_gz = os.path.join(path, file_name, 'eval_results.raw.gz')
            fairness_log = os.path.join(path, file_name, 'eval_results.raw')
            if os.path.exists(fairness_gz):
                with gzip.open(fairness_gz, 'rb') as f:
                    fairness_list = read_fairness(f.readlines())
            elif os.path.exists(fairness_log):
                with open(fairness_log, 'rb') as f:
                    fairness_list = read_fairness(f.readlines())

            with open(os.path.join(path, file_name, 'exp_print.log')) as f:
                F = f.readlines()
                start_time = datetime.strptime(F[0][:19], '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(F[-1][:19], '%Y-%m-%d %H:%M:%S')
                time_dict['tol_time'].append(end_time - start_time)

                train_p = False

                for idx, line in enumerate(F):
                    # Time
                    try:
                        timestamp = datetime.strptime(line[:19],
                                                      '%Y-%m-%d %H:%M:%S')
                    except:
                        continue

                    if "'Role': 'Client #" in line and train_p == False:
                        train_start_time = previous_time
                        train_p = True

                    if "'Role': 'Client #" not in line and train_p == True:
                        train_time = previous_time - train_start_time
                        time_dict['train_time'].append(train_time)
                        train_p = False

                    if 'Starting evaluation' in line:
                        eval_start_time = timestamp
                    if 'Results_raw' in line and 'test' in line:
                        eval_time = timestamp - eval_start_time
                        time_dict['eval_time'].append(eval_time)
                    previous_time = timestamp

                    # Statistics
                    try:
                        results = eval(line.split('INFO: ')[1])
                    except:
                        continue
                    for key in metrics_dict:
                        if results['Role'] in [
                                'Server #', 'Global-Eval-Server #'
                        ] and 'Results_raw' in results.keys():
                            try:
                                metrics_dict[key].append(
                                    results['Results_raw'][key])
                            except KeyError:
                                continue
                        elif 'Results_weighted_avg' not in results:
                            continue
                        else:
                            metrics_dict[key].append(
                                results['Results_weighted_avg'][key])
                if fairness_list and cnt < len(fairness_list):
                    metrics_dict = {**metrics_dict, **fairness_list[cnt]}
                value = [
                    float(file_name.split('_')[i][len(arg):])
                    for i, arg in enumerate(args)
                ]
                df.loc[cnt] = [float(rate)] + value + [{
                    **metrics_dict,
                    **time_dict
                }]
                cnt += 1
    return df
