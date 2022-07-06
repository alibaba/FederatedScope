import datetime
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
        self.model, self.dname, self.algo, self.cost_mode = model, dname, algo, cost_mode
        self.table, self.meta_info = load_data(datadir, model, dname, algo)
        self.eval_freq = self.meta_info['eval_freq']
        super(TabularBenchmark, self).__init__(model, dname, algo, rng,
                                               **kwargs)

    def _check(self, configuration, fidelity):
        for key, value in configuration.items():
            assert value in self.configuration_space[
                key], 'configuration invalid, check `configuration_space` for help.'
        for key, value in fidelity.items():
            assert value in self.fidelity_space[
                key], 'fidelity invalid, check `fidelity_space` for help.'

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
                           configuration,
                           fidelity,
                           key='val_acc',
                           seed=1,
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
        function_value = filterd_result[key][best_round]
        if self._cost(configuration, fidelity, **kwargs):
            cost = self._cost(configuration, fidelity, **kwargs)
        else:
            cost = filterd_result['tol_time']

        return {'function_value': function_value, 'cost': cost}

    def get_configuration_space(self):
        return dict2cfg(self.meta_info['configuration_space'])

    def get_fidelity_space(self):
        return dict2cfg(self.meta_info['fidelity_space'])

    def get_meta_info(self):
        return {
            'model': self.model,
            'dname': self.dname,
            'configuration_space': self.configuration_space,
            'fidelity_space': self.fidelity_space
        }
