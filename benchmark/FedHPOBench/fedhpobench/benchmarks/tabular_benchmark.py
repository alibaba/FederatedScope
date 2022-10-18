import datetime
import logging
import numpy as np

from fedhpob.utils.util import dict2cfg
from fedhpob.utils.tabular_dataloader import load_data
from fedhpob.benchmarks.base_benchmark import BaseBenchmark


class TabularBenchmark(BaseBenchmark):
    def __init__(self,
                 model,
                 dname,
                 algo,
                 datadir='data/tabular_data/',
                 rng=None,
                 cost_mode='estimated',
                 **kwargs):
        self.table, self.meta_info = load_data(datadir, model, dname, algo)
        self.eval_freq = self.meta_info['eval_freq']
        super(TabularBenchmark, self).__init__(model, dname, algo, cost_mode,
                                               rng, **kwargs)

    def _check(self, configuration, fidelity):
        for key, value in configuration.items():
            assert value in self.configuration_space[
                key], 'configuration invalid, check `configuration_space` ' \
                      'for help.'
        for key, value in fidelity.items():
            assert value in self.fidelity_space[
                key], 'fidelity invalid, check `fidelity_space` for help.'

    def objective_function(self,
                           configuration,
                           fidelity,
                           key='val_acc',
                           seed=1,
                           fairness_reg_func=None,
                           fairness_reg_coef=0.0,
                           **kwargs):
        fidelity = self._init_fidelity(fidelity)
        self._check(configuration, fidelity)
        result = self._search(
            {
                'seed': self.rng.randint(seed) %
                len(self.configuration_space['seed']) + 1,
                **configuration
            }, fidelity)
        index = list(result.keys())
        assert len(index) == 1, 'Multiple results.'
        filterd_result = eval(result[index[0]])
        assert key in filterd_result.keys(
        ), f'`key` should be in {filterd_result.keys()}.'
        # Find the best val round.
        val_loss = filterd_result['val_avg_loss']
        best_round = np.argmin(val_loss[:fidelity['round'] + 1])

        # Fairness reg, default is 0.0
        reg_term = 0
        if fairness_reg_func is not None:
            fair_key = key + '_fair'
            try:
                vector_value = filterd_result[fair_key][best_round]
            except KeyError:
                vector_value = None
                logging.WARNING(f'{fair_key} is not in Benchmark.')

            if vector_value is not None:
                reg_term = fairness_reg_func(vector_value) * fairness_reg_coef

        function_value = filterd_result[key][best_round] + reg_term
        if self._cost(configuration, fidelity, **kwargs):
            cost = self._cost(configuration, fidelity, **kwargs)
        else:
            cost = filterd_result['tol_time']

        return {'function_value': function_value, 'cost': cost}

    def get_configuration_space(self, CS=False):
        if not CS:
            return self.meta_info['configuration_space']
        tmp_dict = {}
        for key in self.meta_info['configuration_space']:
            tmp_dict[key] = list(self.meta_info['configuration_space'][key])
        return dict2cfg(tmp_dict)

    def get_fidelity_space(self, CS=False):
        if not CS:
            return self.meta_info['fidelity_space']
        tmp_dict = {}
        for key in self.meta_info['fidelity_space']:
            tmp_dict[key] = list(self.meta_info['fidelity_space'][key])
        return dict2cfg(tmp_dict)

    def get_meta_info(self):
        return {
            'model': self.model,
            'dname': self.dname,
            'configuration_space': self.configuration_space,
            'fidelity_space': self.fidelity_space
        }
