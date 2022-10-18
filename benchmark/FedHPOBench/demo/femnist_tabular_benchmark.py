import logging

from pathlib import Path
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from base_fed_tabular_benchmark import BaseTabularFedHPOBench

__version__ = '0.0.1'

logger = logging.getLogger('FEMNISTabularFed')


class FEMNISTTabularFedHPOBench(BaseTabularFedHPOBench):
    def __init__(self,
                 data_path: Union[str, Path],
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        This is a FL HPO benchmark for 2-layer CNN on FEMNIST with FedAvg
        from paper:
        "FedHPO-Bench: A Benchmark Suite for Federated Hyperparameter
        Optimization",
        url: https://arxiv.org/pdf/2206.03966v4.pdf
        Source: https://github.com/alibaba/FederatedScope/tree/master
        /benchmark/FedHPOB
        Parameters
        ----------
        data_path : str, Path
            Path to Tabular data
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """

        url = "https://federatedscope.oss-cn-beijing.aliyuncs.com" \
              "/fedhpob_cnn_tabular.zip"
        triplets = ('cnn', 'femnist', 'avg')
        client_num = 200
        num_param = 871294
        super(FEMNISTTabularFedHPOBench, self).__init__(data_path,
                                                        url,
                                                        triplets,
                                                        client_num,
                                                        num_param,
                                                        rng=rng)

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
            CS.CategoricalHyperparameter('lr',
                                         choices=[
                                             0.01, 0.01668, 0.02783, 0.04642,
                                             0.07743, 0.12915, 0.21544,
                                             0.35938, 0.59948, 1.0
                                         ]))
        cs.add_hyperparameter(
            CS.CategoricalHyperparameter('wd', choices=[0.0, 0.001, 0.01,
                                                        0.1]))
        cs.add_hyperparameter(
            CS.CategoricalHyperparameter('dropout', choices=[0.0, 0.5]))
        cs.add_hyperparameter(
            CS.CategoricalHyperparameter('batch', choices=[16, 32, 64]))
        cs.add_hyperparameter(
            CS.CategoricalHyperparameter('step', choices=[1, 2, 3, 4]))
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
            CS.CategoricalHyperparameter('sample_client',
                                         choices=[0.2, 0.4, 0.6, 0.8, 1.0],
                                         default_value=1.0)
        ])

        return fidel_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Tabular Benchmarks for FEMNIST FedAvg',
            'references': [
                '@article{Wang2022FedHPOBench,'
                'title   = {FedHPO-Bench: A Benchmark Suite for Federated '
                'Hyperparameter Optimization},'
                'author  = {Zhen Wang and Weirui Kuang and Ce Zhang and '
                'Bolin Ding and Yaliang Li},'
                'journal = {arXiv preprint arXiv:2206.03966},'
                'year    = {2022}}', 'https://arxiv.org/pdf/2206.03966v4.pdf',
                'https://github.com/alibaba/FederatedScope/tree/master'
                '/benchmark/FedHPOB'
            ],
            'code': 'https://github.com/alibaba/FederatedScope/tree/master'
            '/benchmark/FedHPOBench',
        }


if __name__ == '__main__':
    b = FEMNISTTabularFedHPOBench('data', 1)
    config = b.get_configuration_space(seed=1).sample_configuration()
    result_dict = b.objective_function(configuration=config, rng=1)
    print(result_dict)
