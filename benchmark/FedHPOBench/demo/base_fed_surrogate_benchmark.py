import os
import pickle
import logging

from pathlib import Path
from typing import Union, Dict, Tuple

import ConfigSpace as CS
import numpy as np

from base_fed_tabular_benchmark import BaseTabularFedHPOBench

__version__ = '0.0.1'

logger = logging.getLogger('BaseFedHPOBench')


class BaseSurrogateFedHPOBench(BaseTabularFedHPOBench):
    def __init__(self,
                 data_path: Union[str, Path],
                 model_path: Union[str, Path],
                 data_url: str,
                 model_url: str,
                 triplets: Tuple,
                 client_num: int,
                 num_param: int,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        This is a base FL HPO surrogate benchmark from paper:
        "FedHPO-Bench: A Benchmark Suite for Federated Hyperparameter
        Optimization",
        url: https://arxiv.org/pdf/2206.03966v4.pdf
        Source: https://github.com/alibaba/FederatedScope/tree/master
        /benchmark/FedHPOBench
        Parameters
        ----------
        data_path : str, Path
            Path to Tabular data
        data_path : str, Path
            Path to surrogate models
        data_url : download url for raw data
        model_url: download url for surrogate models
        triplets: Tuple, (model, dataset_name, algo)
        client_num: total client_num joining the FL
        num_param: number of model param
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """
        self.target_key = 'val_avg_loss'
        self.surrogate_models = []
        self.info = None
        self.model_path = model_path
        self.model_url = model_url
        super(BaseSurrogateFedHPOBench,
              self).__init__(data_path, data_url, triplets, client_num,
                             num_param, rng)

    def _setup(self):
        super(BaseSurrogateFedHPOBench, self)._setup()
        # Download and extract the model.
        file_list = [f'surrogate_model_{x}.pkl' for x in range(10)]
        root_path = self.download_and_extract(self.model_url, self.model_path,
                                              file_list)
        file_names = os.listdir(root_path)
        model_list = []
        for fname in file_names:
            if not fname.startswith('surrogate_model'):
                continue
            with open(os.path.join(root_path, fname), 'rb') as f:
                model_state = f.read()
                model = pickle.loads(model_state)
                model_list.append(model)
        self.surrogate_models = model_list

    def get_results(self, configuration, fidelity, seed_id):
        return self._make_prediction(configuration, fidelity, seed_id)

    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict,
                                           None] = None,
                           seed_index: Union[int, Tuple, None] = (1, 2, 3),
                           rng: Union[np.random.RandomState, int, None] = None,
                           key: str = 'val_avg_loss',
                           **kwargs) -> Dict:
        assert key == self.target_key, f'The key should be' \
                                       f' {self.target_key}, ' \
                                       f'but get {key}.'
        return super(BaseSurrogateFedHPOBench,
                     self).objective_function(configuration, fidelity,
                                              seed_index, rng, key, **kwargs)

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
