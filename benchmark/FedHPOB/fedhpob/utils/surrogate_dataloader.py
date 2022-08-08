import datetime
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate as sk_cross_validate
from tqdm import tqdm

from fedhpob.utils.tabular_dataloader import load_data


def sampling(X, Y, over_rate=1, down_rate=1.0, cvg_score=0.5):
    rel_score = Y
    over_X = np.repeat(X[rel_score > cvg_score], over_rate, axis=0)
    over_Y = np.repeat(Y[rel_score > cvg_score], over_rate, axis=0)

    mask = np.random.choice(X[rel_score <= cvg_score].shape[0],
                            size=int(X[rel_score <= cvg_score].shape[0] *
                                     down_rate),
                            replace=False)
    down_X = np.array(X[rel_score <= cvg_score])[mask]
    down_Y = np.array(Y[rel_score <= cvg_score])[mask]
    return np.concatenate([over_X, down_X],
                          axis=0), np.concatenate([over_Y, down_Y], axis=0)


def load_surrogate_model(modeldir, model, dname, algo):
    model_list = []
    path = os.path.join(modeldir, model, dname, algo)
    file_names = os.listdir(path)
    for fname in file_names:
        if not fname.startswith('surrogate_model'):
            continue
        with open(os.path.join(path, fname), 'rb') as f:
            model_state = f.read()
            model = pickle.loads(model_state)
            model_list.append(model)

    infofile = os.path.join(path, 'info.pkl')
    with open(infofile, 'rb') as f:
        info = pickle.loads(f.read())

    # TODO: remove X and Y
    X = np.load(os.path.join(path, 'X.npy'))
    Y = np.load(os.path.join(path, 'Y.npy'))

    return model_list, info, X, Y


def build_surrogate_model(datadir, model, dname, algo, key='val_acc'):
    r"""
    from TabularBenchmark to SurrogateBenchmark data format
    """
    table, meta_info = load_data(datadir, model, dname, algo)
    savedir = os.path.join('data/surrogate_model', model, dname, algo)
    os.makedirs(savedir, exist_ok=True)
    # Build data to train the surrogate_model
    X, Y = [], []
    fidelity_space = sorted(['sample_client', 'round'])
    configuration_space = sorted(
        list(set(table.keys()) - {'result', 'seed'} - set(fidelity_space)))

    if not os.path.exists(os.path.join(savedir,
                                       'X.npy')) or not os.path.exists(
                                           os.path.join(savedir, 'Y.npy')):
        print('Building data mat...')
        for idx in tqdm(range(len(table))):
            row = table.iloc[idx]
            x = [row[col]
                 for col in configuration_space] + [row['sample_client']]
            result = eval(row['result'])
            val_loss = result['val_avg_loss']
            for rnd in range(len(val_loss)):
                X.append(x + [rnd * meta_info['eval_freq']])
                best_round = np.argmin(val_loss[:rnd + 1])
                Y.append(result[key][best_round])
        X, Y = np.array(X), np.array(Y)
        np.save(os.path.join(savedir, 'X.npy'), X)
        np.save(os.path.join(savedir, 'Y.npy'), Y)
    else:
        print('Loading cache...')
        X = np.load(os.path.join(savedir, 'X.npy'))
        Y = np.load(os.path.join(savedir, 'Y.npy'))

    new_X, new_Y = sampling(X, Y, over_rate=1, down_rate=1)

    perm = np.random.permutation(np.arange(len(new_Y)))
    new_X, new_Y = new_X[perm], new_Y[perm]

    best_res = -np.inf
    # Ten-fold validation to get ten surrogate_model
    for n_estimators in [10, 20]:
        for max_depth in [10, 15, 20]:
            regr = RandomForestRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth)
            # dict_keys(['fit_time', 'score_time', 'estimator',
            # 'test_score', 'train_score'])
            res = sk_cross_validate(regr,
                                    new_X,
                                    new_Y,
                                    cv=10,
                                    n_jobs=-1,
                                    scoring='neg_mean_absolute_error',
                                    return_estimator=True,
                                    return_train_score=True)
            test_metric = np.mean(res['test_score'])
            train_metric = np.mean(res['train_score'])
            print(f'n_estimators: {n_estimators}, max_depth: {max_depth}, '
                  f'train_metric: {train_metric}, test_metric: {test_metric}')
            if test_metric > best_res:
                best_res = test_metric
                best_models = res['estimator']

    # Save model
    for i, rf in enumerate(best_models):
        file_name = f'surrogate_model_{i}.pkl'
        model_state = pickle.dumps(rf)
        with open(os.path.join(savedir, file_name), 'wb') as f:
            f.write(model_state)

    # Save info
    info = {
        'configuration_space': configuration_space,
        'fidelity_space': fidelity_space
    }
    pkl = pickle.dumps(info)
    with open(os.path.join(savedir, 'info.pkl'), 'wb') as f:
        f.write(pkl)

    return best_models, info, X, Y
