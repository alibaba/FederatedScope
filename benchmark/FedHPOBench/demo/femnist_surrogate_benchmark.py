import logging

from pathlib import Path
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from base_fed_surrogate_benchmark import BaseSurrogateFedHPOBench

__version__ = '0.0.1'

logger = logging.getLogger('FEMNISTabularFed')


class FENISTSurrogateFedHPOBench(BaseSurrogateFedHPOBench):
    def __init__(self,
                 data_path: Union[str, Path],
                 model_path: Union[str, Path],
                 rng: Union[np.random.RandomState, int, None] = None):
        data_url = "https://federatedscope.oss-cn-beijing.aliyuncs.com" \
                    "/fedhpob_cnn_tabular.zip"
        model_url = "https://federatedscope.oss-cn-beijing.aliyuncs.com" \
                    "/fedhpob_cnn_surrogate.zip"
        triplets = ('cnn', 'femnist', 'avg')
        client_num = 200
        num_param = 6603902
        super(FENISTSurrogateFedHPOBench,
              self).__init__(data_path, model_path, data_url, model_url,
                             triplets, client_num, num_param, rng)

    @staticmethod
    def get_configuration_space(
            seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Interface to the get_configuration_space function from the FEMNIST
        Benchmark.
        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.
        Returns
        -------
        CS.ConfigurationSpace -
            Containing the benchmark's hyperparameter
        """

        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter('lr',
                                          lower=1e-2,
                                          upper=1.0,
                                          log=True))
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter('wd',
                                          lower=10e-5,
                                          upper=0.1,
                                          log=True))
        cs.add_hyperparameter(
            CS.UniformFloatHyperparameter('dropout', lower=0, upper=0.5))
        cs.add_hyperparameter(
            CS.UniformIntegerHyperparameter('batch', lower=16, upper=64))
        cs.add_hyperparameter(
            CS.UniformIntegerHyperparameter('step', lower=1, upper=4))
        return cs

    @staticmethod
    def get_fidelity_space(
            seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity
        parameters for
        the FCNetBaseBenchmark
        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace
        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('round',
                                            lower=0,
                                            upper=249,
                                            default_value=249)
        ])
        fidel_space.add_hyperparameters([
            CS.UniformFloatHyperparameter('sample_client',
                                          lower=0,
                                          upper=1.0,
                                          default_value=1.0)
        ])

        return fidel_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Surrogate Benchmarks for FEMNIST FedAvg',
            'references': [
                '@article{Wang2022FedHPOBench,'
                'title   = {FedHPO-Bench: A Benchmark Suite for Federated '
                'Hyperparameter Optimization},'
                'author  = {Zhen Wang and Weirui Kuang and Ce Zhang and '
                'Bolin Ding and Yaliang Li},'
                'journal = {arXiv preprint arXiv:2206.03966},'
                'year    = {2022}}', 'https://arxiv.org/pdf/2206.03966v4.pdf',
                'https://github.com/alibaba/FederatedScope/tree/master'
                '/benchmark/FedHPOBench'
            ],
            'code': 'https://github.com/alibaba/FederatedScope/tree/master'
            '/benchmark/FedHPOBench',
        }


if __name__ == '__main__':
    b = FENISTSurrogateFedHPOBench('data', 'model', 1)
    config = b.get_configuration_space(seed=1).sample_configuration()
    result_dict = b.objective_function(configuration=config, rng=1)
    print(result_dict)
