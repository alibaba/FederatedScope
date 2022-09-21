import os
import ssl
import datetime
import urllib.request
import logging
import zipfile
import pickle

from pathlib import Path
from typing import Union, Dict, Tuple, List

import ConfigSpace as CS
import pandas as pd
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper

__version__ = '0.0.1'

logger = logging.getLogger('BaseFedHPOB')


class BaseFedHPOB(AbstractBenchmark):
    def __init__(self,
                 data_path: Union[str, Path],
                 url: str,
                 triplets: Tuple,
                 client_num: int,
                 num_param: int,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        This is a base FL HPO benchmark from paper:
        "FedHPO-B: A Benchmark Suite for Federated Hyperparameter
        Optimization",
        url: https://arxiv.org/pdf/2206.03966v4.pdf
        Source: https://github.com/alibaba/FederatedScope/tree/master
        /benchmark/FedHPOB
        Parameters
        ----------
        data_path : str, Path
            Path to Tabular data
        url : download url
        triplets: Tuple, (model, dataset_name, algo)
        client_num: total client_num joining the FL
        num_param: number of model param
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """

        self.data_path = data_path
        self.url = url
        self.triplets = triplets
        self.client_num = client_num
        self.num_param = num_param
        self.table, self.info = self._setup()
        self.bandwidth = {
            'client_up': 0.25 * 1024 * 1024 * 8 / 32,
            'client_down': 0.75 * 1024 * 1024 * 8 / 32,
            'server_up': 0.25 * 1024 * 1024 * 8 / 32,
            'server_down': 0.75 * 1024 * 1024 * 8 / 32
        }
        self.server_cmp_cost = 1.0
        super(BaseFedHPOB, self).__init__(rng=rng)

    def _setup(self):
        """ Download and extract the data. """
        file = self.url.rpartition('/')[2]
        file = file if file[0] == '?' else file.split('?')[0]
        path = os.path.join(self.data_path, file)

        root_path = os.path.join(self.data_path, self.triplets[0],
                                 self.triplets[1], self.triplets[2])
        datafile = os.path.join(root_path, 'tabular.csv.gz')
        infofile = os.path.join(root_path, 'info.pkl')

        # Download
        if os.path.exists(path):
            logger.info(f'File {file} exists, use existing file.')
        else:
            logger.info(f'Downloading {self.url}')
            os.makedirs(self.data_path, exist_ok=True)
            ctx = ssl._create_unverified_context()
            data = urllib.request.urlopen(self.url, context=ctx)
            with open(path, 'wb') as f:
                f.write(data.read())

        # Extract
        if not os.path.exists(datafile) and not os.path.exists(infofile):
            with zipfile.ZipFile(path, 'r') as f:
                f.extractall(self.data_path)

        df = pd.read_csv(datafile)
        with open(infofile, 'rb') as f:
            info = pickle.loads(f.read())

        return df, info

    def _get_lambda_from_df(self, configuration, fidelity):
        lambdas = []
        for seed in [0, 1, 2]:
            result = self._search({'seed': seed, **configuration}, fidelity)
            index = list(result.keys())
            filterd_result = eval(result[index[0]])
            c = np.mean(filterd_result['train_time']) + np.mean(
                filterd_result['eval_time'])
            lambdas.append(c.total_seconds())
        return np.mean(lambdas) / float(self.client_num)

    def _cost(self, configuration, fidelity):
        try:
            const = self._get_lambda_from_df(configuration, fidelity)
        except:
            const = 1.0

        cmp_cost = sum([
            1.0 / i for i in range(
                1,
                int(self.client_num * fidelity['sample_client']) + 1)
        ]) * const + self.server_cmp_cost
        cmm_cost = self.num_param / self.bandwidth['client_up'] + max(
            self.client_num * fidelity['sample_client'] * self.num_param /
            self.bandwidth['server_up'],
            self.num_param / self.bandwidth['client_down'])
        return cmp_cost + cmm_cost

    def _search(self, configuration, fidelity):
        # For configuration
        mask = np.array([True] * self.table.shape[0])
        for col in configuration.keys():
            mask *= (self.table[col].values == configuration[col])
        idx = np.where(mask)
        result = self.table.iloc[idx]

        # For fidelity
        mask = np.array([True] * result.shape[0])
        for col in fidelity.keys():
            if col == 'round':
                continue
            mask *= (result[col].values == fidelity[col])
        idx = np.where(mask)
        result = result.iloc[idx]["result"]
        return result

    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict,
                                           None] = None,
                           seed_index: Union[int, Tuple, None] = (1, 2, 3),
                           rng: Union[np.random.RandomState, int, None] = None,
                           key: str = 'val_avg_loss',
                           **kwargs) -> Dict:
        """
        Query the FEMNIST-benchmark using a given configuration and a (
        round, sample_client_rate) (=budget).
        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (
            max) value if None.
        seed_index : int, Tuple, None
            The nas benchmark has for each configuration-budget-pair results
            from 3 different runs with three seed.
            If multiple `seed_id`s are given, the benchmark returns the mean
            over the given runs.
            By default (no parameter is specified) all runs are used.
            A specific run can be chosen by setting the
            `seed_id` to a value from [1, 3].
            When this value is explicitly set to `None`,
            the function will use a random seed.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent
            overfitting on a single seed, it is
            possible to pass a parameter ``rng`` as 'int' or
            'np.random.RandomState' to this
            function. If this parameter is not given, the default random
            state is used.
        key : target key of evaluation metric.
        kwargs
        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
        """
        if fidelity is None:
            fidelity = self.get_fidelity_space().get_default_configuration(
            ).get_dictionary()
        if isinstance(seed_index, int):
            assert 1 <= seed_index <= 3, f'run_index must be in [1, 3], ' \
                                         f'not {seed_index}'
            seed_index = (seed_index, )
        elif isinstance(seed_index, (Tuple, List)):
            assert len(seed_index) != 0, 'run_index must not be empty'
            if len(set(seed_index)) != len(seed_index):
                logger.debug(
                    'There are some values more than once in the run_index. '
                    'We remove the redundant entries.')
            run_index = tuple(set(seed_index))
            assert min(run_index) >= 1 and max(run_index) <= 3, \
                f'all run_index values must be in [0, 3], but were {run_index}'
        elif seed_index is None:
            logger.debug(
                'The seed index is explicitly set to None! A random seed '
                'will be selected.')
            seed_index = tuple(self.rng.choice((1, 2, 3), size=1))
        else:
            raise ValueError(f'run index must be one of Tuple or Int, but was'
                             f' {type(seed_index)}')

        function_values, costs = [], []
        for seed_id in seed_index:
            result = self._search({'seed': seed_id, **configuration}, fidelity)
            index = list(result.keys())
            assert len(index) == 1, 'Multiple results.'
            filterd_result = eval(result[index[0]])
            assert key in filterd_result.keys(
            ), f'`key` should be in {filterd_result.keys()}.'

            # Find the best val round.
            val_loss = filterd_result['val_avg_loss']
            best_round = np.argmin(val_loss[:fidelity['round'] + 1])
            function_value = filterd_result[key][best_round]

            function_values.append(function_value)
            costs.append(self._cost(configuration, fidelity))

        return {
            'function_value': float(np.mean(function_values)),
            'cost': float(sum(costs)),
            'info': {
                f'{key}_per_run': function_values,
                'runtime_per_run': costs,
                'fidelity': fidelity
            },
        }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self,
                                configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, None] = None,
                                rng: Union[np.random.RandomState, int,
                                           None] = None,
                                **kwargs) -> Dict:
        """
        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (
            max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent
            overfitting on a single seed, it is possible to pass a parameter
            ``rng`` as 'int' or 'np.random.RandomState' to this
            function. If this parameter is not given, the default random
            state is used.
        kwargs
        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                valid_rmse_per_run
                runtime_per_run
                fidelity : used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        default_fidelity = self.get_fidelity_space().get_default_configuration(
        ).get_dictionary()
        assert fidelity == default_fidelity, 'Test function works only on ' \
                                             'the highest budget.'
        result = self.objective_function(configuration, default_fidelity)

        return {
            'function_value': float(result['function_value']),
            'cost': float(result['cost']),
            'info': {
                'fidelity': fidelity
            },
        }

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
