import abc
import os
import pickle
import datetime
import numpy as np
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from fedhpob.utils.tabular_dataloader import load_data
from fedhpob.utils.util import disable_fs_logger
from fedhpob.utils.cost_model import get_cost_model


class BaseBenchmark(abc.ABC):
    def __init__(self, model, dname, algo, cost_mode, rng=None, **kwargs):
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
        self.model, self.dname, self.algo, self.cost_mode = model, dname, \
            algo, cost_mode
        # Load data and modify cfg of FS.
        self.cfg = global_cfg.clone()
        self.cfg.set_new_allowed(True)
        filepath = os.path.join('scripts', model, f'{dname}.yaml')
        self.cfg.merge_from_file(filepath)
        self.cfg.data.type = dname
        self.data, modified_cfg = get_data(config=self.cfg.clone())
        self.cfg.merge_from_other_cfg(modified_cfg)
        # Try load time data
        try:
            datadir = os.path.join('data', 'tabular_data')
            self.table, _ = load_data(datadir, model, dname, algo)
        except:
            self.table = None
        disable_fs_logger(self.cfg, True)

    def __call__(self, configuration, fidelity, seed=1, **kwargs):
        return self.objective_function(configuration=configuration,
                                       fidelity=fidelity,
                                       seed=seed,
                                       **kwargs)

    def _check(self, configuration, fidelity):
        pass

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

    def get_lamba_from_df(self, configuration, fidelity):
        if self.table is not None:
            client_num = self.cfg.federate.client_num * \
                       self.cfg.federate.sample_client_rate
            result = self._search({'seed': 0, **configuration}, fidelity)
            index = list(result.keys())
            filterd_result = eval(result[index[0]])
            c = np.mean(filterd_result['train_time']) + np.mean(
                filterd_result['eval_time'])
            return c.total_seconds() / float(client_num)
        else:
            from fedhpob.config import fhb_cfg
            return fhb_cfg.cost.c

    def _cost(self, configuration, fidelity, **kwargs):
        try:
            kwargs['const'] = self.get_lamba_from_df(configuration, fidelity)
        except:
            from fedhpob.config import fhb_cfg
            kwargs['const'] = fhb_cfg.cost.c
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
