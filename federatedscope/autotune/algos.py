import os
import logging
from copy import deepcopy
import threading
from itertools import product

import numpy as np
import torch

from federatedscope.config import cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.worker import Server, Client
from federatedscope.core.fed_runner import FedRunner
from federatedscope.autotune.choice_types import Discrete, Continuous
from federatedscope.autotune.utils import generate_candidates, config2cmdargs, config2str, summarize_hpo_results


def grid_search(search_space, sample_size=None):
    '''To produce a given nunber of configurations from a given (producted) search space with grid search strategy.

    Arguments:
        search_space (dict): key-value pairs corresponding to the fields and choices.
        sample_size (int): the number of candidates to be returned.
    :returns: the sampled configurations to be executed.
    :rtype: list
    '''
    num_axis = len(search_space)
    num_grid = max(1, int(np.exp(np.log(sample_size) / num_axis)))
    for k, v in search_space.items():
        if isinstance(v, Discrete):
            num_grid = min(num_grid, len(v))

    sampled_cands = []
    for tp in product(*[[(k, v) for v in search_space[k].grid(num_grid)]
                        for k in search_space]):
        trial_cfg = dict(tp)
        sampled_cands.append(trial_cfg)
    return sampled_cands


def random_search(search_space, sample_size):
    '''To produce a given nunber of configurations from a given (producted) search space with random search strategy.
    This algorithm is presented in the paper "Random Search for Hyper-Parameter Optimization" which can be found at https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

    Arguments:
        searchs_space (dict): key corresponds to config field, and value corresponds to the choices.
        sample_size (int): the number of candidates to be returned.
    :returns: the sampled configurations to be executed.
    :rtype: list
    '''

    sampled_cands = list()
    for _ in range(sample_size):
        trial_cfg = dict()
        for k, v in search_space.items():
            trial_cfg[k] = v.sample()
        sampled_cands.append(trial_cfg)
    return sampled_cands


class TrialExecutor(threading.Thread):
    """This class is responsible for executing the FL procedure with a given trial configuration in another thread.
    """
    def __init__(self, cfg_idx, signal, returns, trial_config):
        threading.Thread.__init__(self)

        self._idx = cfg_idx
        self._signal = signal
        self._returns = returns
        self._trial_cfg = trial_config

    def run(self):
        setup_seed(self._trial_cfg.seed)
        data, modified_config = get_data(self._trial_cfg)
        Fed_runner = FedRunner(data=data,
                               server_class=Server,
                               client_class=Client,
                               config=modified_config)
        test_results = Fed_runner.run()
        key1, key2 = self._trial_cfg.hpo.metric.split('.')
        self._returns['perf'] = test_results[key1][key2]
        self._returns['cfg_idx'] = self._idx
        self._signal.set()


def get_scheduler(raw_search_space):
    if cfg.hpo.scheduler == 'bruteforce':
        scheduler = BruteForce(raw_search_space)
    elif cfg.hpo.scheduler == 'sha':
        scheduler = SuccessiveHalvingAlgo(raw_search_space)
    elif cfg.hpo.scheduler == 'pbt':
        scheduler = PBT(raw_search_space)
    return scheduler


class Scheduler(object):
    """The base class for describing HPO algorithms
    """
    def __init__(self, raw_search_space):
        """
            Arguments:
                raw_search_space (dict): each key-value pair corresponds to a field and its choices.
        """
        self._init_configs = self._setup(raw_search_space)
        logging.info(self._init_configs)

    def _setup(self):
        """Record the search space and prepare the initial configurations.
        """
        raise NotImplementedError

    def _evaluate(self, configs):
        """To evaluate (i.e., conduct the FL procedure) for a given collection of configurations.
        """
        raise NotImplementedError

    def optimize(self):
        """To optimize the hyperparameters, that is, executing the HPO algorithm and then returning the results.
        """
        raise NotImplementedError


class BruteForce(Scheduler):
    """To attempt a collection of configurations exhaustively.
    """
    def _setup(self, raw_search_space):
        self._original_search_space = raw_search_space

        if cfg.hpo.init_strategy == 'grid':
            init_configs = grid_search(raw_search_space, cfg.hpo.init_cand_num)
        elif cfg.hpo.init_strategy == 'random':
            init_configs = random_search(raw_search_space,
                                         cfg.hpo.init_cand_num)
        else:
            raise ValueError(
                "BruteForce needs to use random/grid search to pick {} configs from the search space as initial candidates, but `{}` is specified as `hpo.init_strategy`"
                .format(cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num,
                        cfg.hpo.init_strategy))

        return init_configs

    def _evaluate(self, configs):
        device_flags = [
            threading.Event() for _ in range(torch.cuda.device_count())
        ]
        logging.info("Conduct HPO with {} devices in-parallel".format(
            len(device_flags)))
        for i in range(len(device_flags)):
            device_flags[i].set()
        threads = [None for _ in range(len(device_flags))]
        thread_results = [dict() for _ in range(len(device_flags))]

        def pick_device():
            cur_idx = 0
            while True:
                if device_flags[cur_idx].is_set():
                    return cur_idx
                cur_idx = (cur_idx + 1) % len(device_flags)

        perfs = [None for _ in range(len(configs))]
        plots = []
        last_plot, consumed_bgt = 0, 0
        for i, config in enumerate(configs):
            available_device = pick_device()
            if threads[available_device] is not None:
                threads[available_device].join()
                completed_trial_results = thread_results[available_device]
                if len(completed_trial_results) > 0:
                    cfg_idx = completed_trial_results['cfg_idx']
                    perfs[cfg_idx] = completed_trial_results['perf']
                    # update the plots
                    consumed_bgt += configs[cfg_idx][
                        'federate.total_round_num'] if 'federate.total_round_num' in configs[
                            cfg_idx] else cfg.federate.total_round_num
                    if consumed_bgt - cfg.hpo.plot_interval >= last_plot:
                        plots.append(
                            max(
                                completed_trial_results['perf'],
                                max(plots)
                                if plots else completed_trial_results['perf']
                            ) if cfg.hpo.larger_better else min(
                                completed_trial_results['perf'],
                                min(plots)
                                if plots else completed_trial_results['perf']))
                        last_plots = consumed_bgt
            device_flags[available_device].clear()
            thread_results[available_device] = dict()

            trial_cfg = cfg.clone()
            trial_cfg.merge_from_list(config2cmdargs(config))
            trial_cfg.merge_from_list(['device', available_device])
            trial = TrialExecutor(i, device_flags[available_device],
                                  thread_results[available_device], trial_cfg)
            trial.start()
            threads[available_device] = trial

        for i in range(len(device_flags)):
            if threads[i] is not None:
                threads[i].join()
                if len(thread_results[i]) > 0:
                    completed_trial_results = thread_results[i]
                    cfg_idx = completed_trial_results['cfg_idx']
                    perfs[cfg_idx] = float(completed_trial_results['perf'])
                    # update the plots
                    consumed_bgt += configs[cfg_idx][
                        'federate.total_round_num'] if 'federate.total_round_num' in configs[
                            cfg_idx] else cfg.federate.total_round_num
                    if consumed_bgt - cfg.hpo.plot_interval >= last_plot:
                        plots.append(
                            max(
                                completed_trial_results['perf'],
                                max(plots)
                                if plots else completed_trial_results['perf']
                            ) if cfg.hpo.larger_better else min(
                                completed_trial_results['perf'],
                                min(plots)
                                if plots else completed_trial_results['perf']))
                        last_plots = consumed_bgt

        return perfs, plots

    def optimize(self):
        perfs, plots = self._evaluate(self._init_configs)
        results = summarize_hpo_results(
            self._init_configs,
            perfs,
            white_list=set(self._original_search_space.keys()),
            desc=cfg.hpo.larger_better)
        logging.info(
            "====================================== Final ========================================"
        )
        logging.info("\n{}".format(results))
        logging.info(
            "====================================================================================="
        )
        logging.info("The performance changes as {}".format(plots))
        return results


class IterativeScheduler(BruteForce):
    """The base class for HPO algorithms that divide the whole optimization procedure into iterations.
    """
    def _setup(self, raw_search_space):
        self._original_search_space = raw_search_space
        self._stage = 0
        self._accum_plots = []
        return []

    def _stop_criterion(self, configs, last_results):
        """To determine whether the algorithm should be terminated.

        Arguments:
            configs (list): each element is a trial configuration.
            last_results (DataFrame): each row corresponds to a specific configuration as well as its latest performance.
        :returns: whether to terminate.
        :rtype: bool
        """
        raise NotImplementedError

    def _iteration(self, configs):
        """To evaluate the given collection of configurations at this stage.

        Arguments:
            configs (list): each element is a trial configuration.
        :returns: the performances of the given configurations.
        :rtype: list
        """

        perfs, plots = self._evaluate(configs)
        self._accum_plots.append(plots)
        return perfs

    def _generate_next_population(self, configs, perfs):
        """To generate the configurations for the next stage.

        Arguments:
            configs (list): the configurations of last stage.
            perfs (list): their corresponding performances.
        :returns: configuration for the next stage.
        :rtype: list
        """

        raise NotImplementedError

    def optimize(self):
        current_configs = deepcopy(self._init_configs)
        last_results = None
        while not self._stop_criterion(current_configs, last_results):
            current_perfs = self._iteration(current_configs)
            last_results = summarize_hpo_results(
                current_configs,
                current_perfs,
                white_list=set(self._original_search_space.keys()),
                desc=cfg.hpo.larger_better)
            self._stage += 1
            logging.info(
                "====================================== Stage{} ========================================"
                .format(self._stage))
            logging.info("\n{}".format(last_results))
            logging.info(
                "======================================================================================="
            )
            current_configs = self._generate_next_population(
                current_configs, current_perfs)
        # output the performance v.s. consumed budget
        logging.info("Performance changes as {}".format(
            [elem for stg_plts in self._accum_plots for elem in stg_plts]))
        return current_configs


class SuccessiveHalvingAlgo(IterativeScheduler):
    """Successive Halving Algorithm (SHA) (also known as Hyperband where the full paper "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" can be found at https://jmlr.org/papers/volume18/16-558/16-558.pdf) tailored to FL setting, where, in each iteration, just a limited number of communication rounds are allowed for each trial.
    """
    def _setup(self, raw_search_space):
        _ = super(SuccessiveHalvingAlgo, self)._setup(raw_search_space)

        if cfg.hpo.init_strategy == 'random':
            init_configs = random_search(
                raw_search_space,
                sample_size=cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num)
        elif cfg.hpo.init_strategy == 'grid':
            init_configs = grid_search(
                raw_search_space,
                sample_size=cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num)
        else:
            raise ValueError(
                "SHA needs to use random/grid search to pick {} configs from the search space as initial candidates, but `{}` is specified as `hpo.init_strategy`"
                .format(cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num,
                        cfg.hpo.init_strategy))

        for trial_cfg in init_configs:
            trial_cfg['federate.save_to'] = os.path.join(
                cfg.hpo.working_folder, "{}.pth".format(config2str(trial_cfg)))

        if cfg.hpo.sha.budgets:
            for trial_cfg in init_configs:
                trial_cfg['federate.total_round_num'] = cfg.hpo.sha.budgets[
                    self._stage]
                trial_cfg['eval.freq'] = cfg.hpo.sha.budgets[self._stage]

        return init_configs

    def _stop_criterion(self, configs, last_results):
        return len(configs) <= 1

    def _generate_next_population(self, configs, perfs):
        indices = [(i, val) for i, val in enumerate(perfs)]
        indices.sort(key=lambda x: x[1], reverse=cfg.hpo.larger_better)
        next_population = [
            configs[tp[0]]
            for tp in indices[:len(indices) // cfg.hpo.sha.elim_rate]
        ]

        for trial_cfg in next_population:
            if 'federate.restore_from' not in trial_cfg:
                trial_cfg['federate.restore_from'] = trial_cfg[
                    'federate.save_to']
            if cfg.hpo.sha.budgets and self._stage < len(cfg.hpo.sha.budgets):
                trial_cfg['federate.total_round_num'] = cfg.hpo.sha.budgets[
                    self._stage]
                trial_cfg['eval.freq'] = cfg.hpo.sha.budgets[self._stage]

        return next_population


class PBT(IterativeScheduler):
    """Population-based training (the full paper "Population Based Training of Neural Networks" can be found at https://arxiv.org/abs/1711.09846)  tailored to FL setting, where, in each iteration, just a limited number of communication rounds are allowed for each trial (We will provide the asynchornous version later).
    """
    def _setup(self, raw_search_space):
        _ = super(PBT, self)._setup(raw_search_space)

        if cfg.hpo.init_strategy == 'random':
            init_configs = random_search(
                raw_search_space,
                sample_size=cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num)
        elif cfg.hpo.init_strategy == 'grid':
            init_configs = grid_search(
                raw_search_space,
                sample_size=cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num)
        else:
            raise ValueError(
                "SHA needs to use random/grid search to pick {} configs from the search space as initial candidates, but `{}` is specified as `hpo.init_strategy`"
                .format(cfg.hpo.sha.elim_rate**cfg.hpo.sha.elim_round_num,
                        cfg.hpo.init_strategy))

        for trial_cfg in init_configs:
            trial_cfg['federate.save_to'] = os.path.join(
                cfg.hpo.working_folder, "{}.pth".format(config2str(trial_cfg)))

        return init_configs

    def _stop_criterion(self, configs, last_results):
        if last_results is not None:
            if (cfg.hpo.larger_better and last_results.iloc[0]['performance']
                    >= cfg.hpo.pbt.perf_threshold) or (
                        (not cfg.hpo.larger_better)
                        and last_results.iloc[0]['performance'] <=
                        cfg.hpo.pbt.perf_threshold):
                return True
        return self._stage >= cfg.hpo.pbt.max_stage

    def _generate_next_population(self, configs, perfs):
        next_generation = []
        for i in range(len(configs)):
            new_cfg = deepcopy(configs[i])
            # exploit
            j = np.random.randint(len(configs))
            if i != j and ((cfg.hpo.larger_better and perfs[j] > perfs[i]) or (
                (not cfg.hpo.larger_better) and perfs[j] < perfs[i])):
                new_cfg['federate.restore_from'] = configs[j][
                    'federate.save_to']
                # explore
                for k in new_cfg:
                    if isinstance(new_cfg[k], float):
                        # according to the exploration strategy of the PBT paper
                        new_cfg[k] *= float(np.random.choice([0.8, 1.2]))
            else:
                new_cfg['federate.restore_from'] = configs[i][
                    'federate.save_to']

            # update save path
            tmp_cfg = dict()
            for k in new_cfg:
                if k in self._original_search_space:
                    tmp_cfg[k] = new_cfg[k]
            new_cfg['federate.save_to'] = os.path.join(
                cfg.hpo.working_folder, "{}.pth".format(config2str(tmp_cfg)))

            next_generation.append(new_cfg)

        return next_generation
