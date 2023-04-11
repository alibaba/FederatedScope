import abc
import os
import numpy as np
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from fedhpob.utils.util import disable_fs_logger
from fedhpob.utils.cost_model import get_cost_model


class BaseBenchmark(abc.ABC):
    def __init__(self, model, dname, algo, rng=None, **kwargs):
        """

        :param rng:
        :param kwargs:
        """
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()
        self.configuration_space = self.get_configuration_space()
        self.fidelity_space = self.get_fidelity_space()

        # Load data and modify cfg of FS.
        self.cfg = global_cfg.clone()
        filepath = os.path.join('scripts', model, f'{dname}.yaml')
        self.cfg.merge_from_file(filepath)
        self.cfg.data.type = dname
        self.data, modified_cfg = get_data(config=self.cfg.clone())
        self.cfg.merge_from_other_cfg(modified_cfg)
        disable_fs_logger(self.cfg, True)

    def __call__(self, configuration, fidelity, seed=1, **kwargs):
        return self.objective_function(configuration=configuration,
                                       fidelity=fidelity,
                                       seed=seed,
                                       **kwargs)

    def _check(self, configuration, fidelity):
        pass

    def _cost(self, configuration, fidelity, **kwargs):
        cost_model = get_cost_model(mode=self.cost_mode)
        t = cost_model(self.cfg, configuration, fidelity, self.data, **kwargs)
        return t

    def _init_fidelity(self, fidelity):
        if not fidelity:
            fidelity = {
                'sample_client': 1.0,
                'round': self.get_fidelity_space()['round'][-1] //
                self.eval_freq
            }
        elif 'round' not in fidelity:
            fidelity['round'] = self.get_fidelity_space(
            )['round'][-1] // self.eval_freq
        return fidelity

    @abc.abstractmethod
    def objective_function(self, configuration, fidelity, seed):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_configuration_space(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_fidelity_space(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_meta_info(self):
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.get_meta_info()})'
