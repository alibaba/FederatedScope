import os
import pickle
import logging

from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate as sk_cross_validate

from base_fed_tabular_benchmark import BaseTabularFedHPOBench

__version__ = '0.0.1'

logger = logging.getLogger('BaseFedHPOBench')


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


class BaseSurrogateFedHPOBench(BaseTabularFedHPOBench):
    target_key = 'val_avg_loss'
    surrogate_models = []
    info = []

    def _setup(self):
        super(BaseSurrogateFedHPOBench, self)._setup()
        self.build_surrogate_model(self.target_key)

    def build_surrogate_model(self, key='val_avg_loss'):
        root_path = os.path.join(self.data_path, self.triplets[0],
                                 self.triplets[1], self.triplets[2])
        os.makedirs(root_path, exist_ok=True)
        X, Y = [], []
        fidelity_space = sorted(['sample_client', 'round'])
        configuration_space = sorted(
            list(set(self.table.keys()) - {'result', 'seed'} - set(
                fidelity_space)))

        if not os.path.exists(os.path.join(root_path,
                                           'X.npy')) or not os.path.exists(
            os.path.join(root_path, 'Y.npy')):
            print('Building data mat...')
            for idx in range(len(root_path)):
                row = root_path.iloc[idx]
                x = [row[col]
                     for col in configuration_space] + [row['sample_client']]
                result = eval(row['result'])
                val_loss = result['val_avg_loss']
                for rnd in range(len(val_loss)):
                    X.append(x + [rnd * self.info['eval_freq']])
                    best_round = np.argmin(val_loss[:rnd + 1])
                    Y.append(result[key][best_round])
            X, Y = np.array(X), np.array(Y)
            np.save(os.path.join(root_path, 'X.npy'), X)
            np.save(os.path.join(root_path, 'Y.npy'), Y)
        else:
            print('Loading cache...')
            X = np.load(os.path.join(root_path, 'X.npy'))
            Y = np.load(os.path.join(root_path, 'Y.npy'))

        new_X, new_Y = sampling(X, Y, over_rate=1, down_rate=1)

        perm = np.random.permutation(np.arange(len(new_Y)))
        new_X, new_Y = new_X[perm], new_Y[perm]

        best_res = -np.inf
        # Ten-fold validation to get ten surrogate_model
        for n_estimators in [10, 20]:
            for max_depth in [10, 15, 20]:
                regr = RandomForestRegressor(n_estimators=n_estimators,
                                             max_depth=max_depth)
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
            with open(os.path.join(root_path, file_name), 'wb') as f:
                f.write(model_state)

        # Save info
        info = {
            'configuration_space': configuration_space,
            'fidelity_space': fidelity_space
        }
        pkl = pickle.dumps(info)
        with open(os.path.join(root_path, 'info.pkl'), 'wb') as f:
            f.write(pkl)
        self.surrogate_models = best_models
        self.info = info

    def get_results(self, configuration, fidelity, seed_id):
        return self._make_prediction(configuration, fidelity, seed_id)

    def _make_prediction(self, configuration, fidelity, seed_id):
        model = self.surrogate_models[seed_id % len(self.surrogate_models)]
        x_in = []
        cfg_keys = sorted(self.configuration_space)
        fid_keys = sorted(self.fidelity_space)
        for key in cfg_keys:
            x_in.append(configuration[key])
        for key in fid_keys:
            x_in.append(fidelity[key])
        return model.predict([x_in])[0]

    @staticmethod
    def get_configuration_space(
            seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @staticmethod
    def get_fidelity_space(
            seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @staticmethod
    def get_meta_information() -> Dict:
        raise NotImplementedError

