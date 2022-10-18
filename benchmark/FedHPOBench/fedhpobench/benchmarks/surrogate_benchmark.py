import os

from fedhpob.benchmarks.base_benchmark import BaseBenchmark
from fedhpob.utils.surrogate_dataloader import build_surrogate_model, \
    load_surrogate_model


class SurrogateBenchmark(BaseBenchmark):
    def __init__(self,
                 model,
                 dname,
                 algo,
                 modeldir='data/surrogate_model/',
                 datadir='data/tabular_data/',
                 rng=None,
                 cost_mode='estimated',
                 **kwargs):
        self.model, self.dname, self.algo, self.cost_mode = model, dname, \
                                                            algo, cost_mode
        assert datadir or modeldir, 'Please provide at least one of ' \
                                    '`datadir` and `modeldir`.'
        if not os.path.exists(os.path.join(modeldir, model, dname, algo)):
            self.surrogate_models, self.meta_info, self.X, self.Y = \
                build_surrogate_model(datadir, model, dname, algo)
        else:
            self.surrogate_models, self.meta_info, self.X, self.Y = \
                load_surrogate_model(modeldir, model, dname, algo)
        super(SurrogateBenchmark, self).__init__(model, dname, algo, cost_mode,
                                                 rng, **kwargs)

    def _check(self, configuration, fidelity):
        for key in configuration:
            assert key in self.configuration_space, 'configuration invalid, ' \
                                                    'check ' \
                                                    '`configuration_space` ' \
                                                    'for help.'
        for key in fidelity:
            assert key in self.fidelity_space, 'fidelity invalid, ' \
                                               'check `fidelity_space` for ' \
                                               'help.'

    def _make_prediction(self, configuration, fidelity, seed):
        model = self.surrogate_models[self.rng.randint(seed) %
                                      len(self.surrogate_models)]
        x_in = []
        cfg_keys = sorted(self.configuration_space)
        fid_keys = sorted(self.fidelity_space)
        for key in cfg_keys:
            x_in.append(configuration[key])
        for key in fid_keys:
            x_in.append(fidelity[key])
        return model.predict([x_in])[0]

    # noinspection DuplicatedCode
    def objective_function(self,
                           configuration,
                           fidelity=None,
                           seed=1,
                           **kwargs):
        fidelity = self._init_fidelity(fidelity)
        self._check(configuration, fidelity)
        return {
            'function_value': self._make_prediction(configuration, fidelity,
                                                    seed),
            'cost': self._cost(configuration, fidelity, **kwargs)
        }

    def get_configuration_space(self):
        new_list = []
        for i in self.meta_info['configuration_space']:
            if i == 'batch':
                new_list.append('batch_size')
            else:
                new_list.append(i)

        return new_list

    def get_fidelity_space(self):
        return self.meta_info['fidelity_space']

    def get_meta_info(self):
        return {
            'model': self.model,
            'dname': self.dname,
            'configuration_space': self.configuration_space,
            'fidelity_space': self.fidelity_space
        }
