import os
import logging
from copy import deepcopy
from contextlib import redirect_stdout
import threading
import math

import ConfigSpace as CS
from yacs.config import CfgNode as CN
import yaml
import numpy as np

from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from federatedscope.core.fed_runner import FedRunner
from federatedscope.autotune.utils import parse_search_space, \
    config2cmdargs, config2str, summarize_hpo_results

logger = logging.getLogger(__name__)


def make_trial(trial_cfg):
    setup_seed(trial_cfg.seed)
    data, modified_config = get_data(config=trial_cfg.clone())
    trial_cfg.merge_from_other_cfg(modified_config)
    trial_cfg.freeze()
    # TODO: enable client-wise configuration
    Fed_runner = FedRunner(data=data,
                           server_class=get_server_cls(trial_cfg),
                           client_class=get_client_cls(trial_cfg),
                           config=trial_cfg.clone())
    results = Fed_runner.run()
    key1, key2 = trial_cfg.hpo.metric.split('.')
    return results[key1][key2]


class TrialExecutor(threading.Thread):
    """This class is responsible for executing the FL procedure with
    a given trial configuration in another thread.
    """
    def __init__(self, cfg_idx, signal, returns, trial_config):
        threading.Thread.__init__(self)

        self._idx = cfg_idx
        self._signal = signal
        self._returns = returns
        self._trial_cfg = trial_config

    def run(self):
        setup_seed(self._trial_cfg.seed)
        data, modified_config = get_data(config=self._trial_cfg.clone())
        self._trial_cfg.merge_from_other_cfg(modified_config)
        self._trial_cfg.freeze()
        # TODO: enable client-wise configuration
        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(self._trial_cfg),
                               client_class=get_client_cls(self._trial_cfg),
                               config=self._trial_cfg.clone())
        results = Fed_runner.run()
        key1, key2 = self._trial_cfg.hpo.metric.split('.')
        self._returns['perf'] = results[key1][key2]
        self._returns['cfg_idx'] = self._idx
        self._signal.set()


def get_scheduler(init_cfg):
    """To instantiate an scheduler object for conducting HPO
    Arguments:
        init_cfg (yacs.Node): configuration.
    """

    if init_cfg.hpo.scheduler == 'rs':
        scheduler = ModelFreeBase(init_cfg)
    elif init_cfg.hpo.scheduler == 'sha':
        scheduler = SuccessiveHalvingAlgo(init_cfg)
    # elif init_cfg.hpo.scheduler == 'pbt':
    #     scheduler = PBT(init_cfg)
    elif init_cfg.hpo.scheduler == 'wrap_sha':
        scheduler = SHAWrapFedex(init_cfg)
    return scheduler


class Scheduler(object):
    """The base class for describing HPO algorithms
    """
    def __init__(self, cfg):
        """
            Arguments:
                cfg (yacs.Node): dict like object, where each key-value pair
                corresponds to a field and its choices.
        """

        self._cfg = cfg
        self._search_space = parse_search_space(self._cfg.hpo.ss)

        self._init_configs = self._setup()

        logger.info(self._init_configs)

    def _setup(self):
        """Prepare the initial configurations based on the search space.
        """
        raise NotImplementedError

    def _evaluate(self, configs):
        """To evaluate (i.e., conduct the FL procedure) for a given
        collection of configurations.
        """
        raise NotImplementedError

    def optimize(self):
        """To optimize the hyperparameters, that is, executing the HPO
        algorithm and then returning the results.
        """
        raise NotImplementedError


class ModelFreeBase(Scheduler):
    """To attempt a collection of configurations exhaustively.
    """
    def _setup(self):
        self._search_space.seed(self._cfg.seed + 19)
        return [
            cfg.get_dictionary()
            for cfg in self._search_space.sample_configuration(
                size=self._cfg.hpo.init_cand_num)
        ]

    def _evaluate(self, configs):
        if self._cfg.hpo.num_workers:
            # execute FL in parallel by multi-threading
            flags = [
                threading.Event() for _ in range(self._cfg.hpo.num_workers)
            ]
            for i in range(len(flags)):
                flags[i].set()
            threads = [None for _ in range(len(flags))]
            thread_results = [dict() for _ in range(len(flags))]

            perfs = [None for _ in range(len(configs))]
            for i, config in enumerate(configs):
                available_worker = 0
                while not flags[available_worker].is_set():
                    available_worker = (available_worker + 1) % len(threads)
                if thread_results[available_worker]:
                    completed_trial_results = thread_results[available_worker]
                    cfg_idx = completed_trial_results['cfg_idx']
                    perfs[cfg_idx] = completed_trial_results['perf']
                    logger.info(
                        "Evaluate the {}-th config {} and get performance {}".
                        format(cfg_idx, configs[cfg_idx], perfs[cfg_idx]))
                    thread_results[available_worker].clear()

                trial_cfg = self._cfg.clone()
                trial_cfg.merge_from_list(config2cmdargs(config))
                flags[available_worker].clear()
                trial = TrialExecutor(i, flags[available_worker],
                                      thread_results[available_worker],
                                      trial_cfg)
                trial.start()
                threads[available_worker] = trial

            for i in range(len(flags)):
                if not flags[i].is_set():
                    threads[i].join()
            for i in range(len(thread_results)):
                if thread_results[i]:
                    completed_trial_results = thread_results[i]
                    cfg_idx = completed_trial_results['cfg_idx']
                    perfs[cfg_idx] = completed_trial_results['perf']
                    logger.info(
                        "Evaluate the {}-th config {} and get performance {}".
                        format(cfg_idx, configs[cfg_idx], perfs[cfg_idx]))
                    thread_results[i].clear()

        else:
            perfs = [None] * len(configs)
            for i, config in enumerate(configs):
                trial_cfg = self._cfg.clone()
                trial_cfg.merge_from_list(config2cmdargs(config))
                perfs[i] = make_trial(trial_cfg)
                logger.info(
                    "Evaluate the {}-th config {} and get performance {}".
                    format(i, config, perfs[i]))

        return perfs

    def optimize(self):
        perfs = self._evaluate(self._init_configs)

        results = summarize_hpo_results(self._init_configs,
                                        perfs,
                                        white_list=set(
                                            self._search_space.keys()),
                                        desc=self._cfg.hpo.larger_better)
        logger.info(
            "========================== HPO Final ==========================")
        logger.info("\n{}".format(results))
        logger.info("====================================================")

        return results


class IterativeScheduler(ModelFreeBase):
    """The base class for HPO algorithms that divide the whole optimization
    procedure into iterations.
    """
    def _setup(self):
        self._stage = 0
        return super(IterativeScheduler, self)._setup()

    def _stop_criterion(self, configs, last_results):
        """To determine whether the algorithm should be terminated.

        Arguments:
            configs (list): each element is a trial configuration.
            last_results (DataFrame): each row corresponds to a specific
            configuration as well as its latest performance.
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

        perfs = self._evaluate(configs)
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
                white_list=set(self._search_space.keys()),
                desc=self._cfg.hpo.larger_better)
            self._stage += 1
            logger.info(
                "========================== Stage{} =========================="
                .format(self._stage))
            logger.info("\n{}".format(last_results))
            logger.info("====================================================")
            current_configs = self._generate_next_population(
                current_configs, current_perfs)

        return current_configs


class SuccessiveHalvingAlgo(IterativeScheduler):
    """Successive Halving Algorithm (SHA) tailored to FL setting, where,
    in each iteration, just a limited number of communication rounds are
    allowed for each trial.
    """
    def _setup(self):
        init_configs = super(SuccessiveHalvingAlgo, self)._setup()

        for trial_cfg in init_configs:
            trial_cfg['federate.save_to'] = os.path.join(
                self._cfg.hpo.working_folder,
                "{}.pth".format(config2str(trial_cfg)))

        if self._cfg.hpo.sha.budgets:
            for trial_cfg in init_configs:
                trial_cfg[
                    'federate.total_round_num'] = self._cfg.hpo.sha.budgets[
                        self._stage]
                trial_cfg['eval.freq'] = self._cfg.hpo.sha.budgets[self._stage]

        return init_configs

    def _stop_criterion(self, configs, last_results):
        return len(configs) <= 1

    def _generate_next_population(self, configs, perfs):
        indices = [(i, val) for i, val in enumerate(perfs)]
        indices.sort(key=lambda x: x[1], reverse=self._cfg.hpo.larger_better)
        next_population = [
            configs[tp[0]] for tp in
            indices[:math.
                    ceil(float(len(indices)) / self._cfg.hpo.sha.elim_rate)]
        ]

        for trial_cfg in next_population:
            if 'federate.restore_from' not in trial_cfg:
                trial_cfg['federate.restore_from'] = trial_cfg[
                    'federate.save_to']
            if self._cfg.hpo.sha.budgets and self._stage < len(
                    self._cfg.hpo.sha.budgets):
                trial_cfg[
                    'federate.total_round_num'] = self._cfg.hpo.sha.budgets[
                        self._stage]
                trial_cfg['eval.freq'] = self._cfg.hpo.sha.budgets[self._stage]

        return next_population


class SHAWrapFedex(SuccessiveHalvingAlgo):
    """This SHA is customized as a wrapper for FedEx algorithm."""
    def _make_local_perturbation(self, config):
        neighbor = dict()
        for k in config:
            if 'fedex' in k or 'fedopt' in k or k in [
                    'federate.save_to', 'federate.total_round_num', 'eval.freq'
            ]:
                # a workaround
                continue
            hyper = self._search_space.get(k)
            if isinstance(hyper, CS.UniformFloatHyperparameter):
                lb, ub = hyper.lower, hyper.upper
                diameter = self._cfg.hpo.table.eps * (ub - lb)
                new_val = (config[k] -
                           0.5 * diameter) + np.random.uniform() * diameter
                neighbor[k] = float(np.clip(new_val, lb, ub))
            elif isinstance(hyper, CS.UniformIntegerHyperparameter):
                lb, ub = hyper.lower, hyper.upper
                diameter = self._cfg.hpo.table.eps * (ub - lb)
                new_val = round(
                    float((config[k] - 0.5 * diameter) +
                          np.random.uniform() * diameter))
                neighbor[k] = int(np.clip(new_val, lb, ub))
            elif isinstance(hyper, CS.CategoricalHyperparameter):
                if len(hyper.choices) == 1:
                    neighbor[k] = config[k]
                else:
                    threshold = self._cfg.hpo.table.eps * len(
                        hyper.choices) / (len(hyper.choices) - 1)
                    rn = np.random.uniform()
                    new_val = np.random.choice(
                        hyper.choices) if rn <= threshold else config[k]
                    if type(new_val) in [np.int32, np.int64]:
                        neighbor[k] = int(new_val)
                    elif type(new_val) in [np.float32, np.float64]:
                        neighbor[k] = float(new_val)
                    else:
                        neighbor[k] = str(new_val)
            else:
                raise TypeError("Value of {} has an invalid type {}".format(
                    k, type(config[k])))

        return neighbor

    def _setup(self):
        # self._cache_yaml()
        init_configs = super(SHAWrapFedex, self)._setup()
        new_init_configs = []
        for idx, trial_cfg in enumerate(init_configs):
            arms = dict(("arm{}".format(1 + j),
                         self._make_local_perturbation(trial_cfg))
                        for j in range(self._cfg.hpo.table.num - 1))
            arms['arm0'] = dict(
                (k, v) for k, v in trial_cfg.items() if k in arms['arm1'])
            with open(
                    os.path.join(self._cfg.hpo.working_folder,
                                 f'{idx}_tmp_grid_search_space.yaml'),
                    'w') as f:
                yaml.dump(arms, f)
            new_trial_cfg = dict()
            for k in trial_cfg:
                if k not in arms['arm0']:
                    new_trial_cfg[k] = trial_cfg[k]
            new_trial_cfg['hpo.table.idx'] = idx
            new_trial_cfg['hpo.fedex.ss'] = os.path.join(
                self._cfg.hpo.working_folder,
                f"{new_trial_cfg['hpo.table.idx']}_tmp_grid_search_space.yaml")
            new_trial_cfg['federate.save_to'] = os.path.join(
                self._cfg.hpo.working_folder, "idx_{}.pth".format(idx))
            new_init_configs.append(new_trial_cfg)

        self._search_space.add_hyperparameter(
            CS.CategoricalHyperparameter("hpo.table.idx",
                                         choices=list(
                                             range(len(new_init_configs)))))

        return new_init_configs


# TODO: refactor PBT to enable async parallel
# class PBT(IterativeScheduler):
#    """Population-based training (the full paper "Population Based Training
#    of Neural Networks" can be found at https://arxiv.org/abs/1711.09846)
#    tailored to FL setting, where, in each iteration, just a limited number
#    of communication rounds are allowed for each trial (We will provide the
#    asynchornous version later).
#    """
#    def _setup(self, raw_search_space):
#        _ = super(PBT, self)._setup(raw_search_space)
#
#        if global_cfg.hpo.init_strategy == 'random':
#            init_configs = random_search(
#                raw_search_space,
#                sample_size=global_cfg.hpo.sha.elim_rate**
#                global_cfg.hpo.sha.elim_round_num)
#        elif global_cfg.hpo.init_strategy == 'grid':
#            init_configs = grid_search(raw_search_space, \
#                sample_size=global_cfg.hpo.sha.elim_rate \
#                **global_cfg.hpo.sha.elim_round_num)
#        else:
#            raise ValueError(
#                "SHA needs to use random/grid search to pick {} configs
#                from the search space as initial candidates, but `{}` is
#                specified as `hpo.init_strategy`"
#                .format(
#                    global_cfg.hpo.sha.elim_rate**
#                    global_cfg.hpo.sha.elim_round_num,
#                    global_cfg.hpo.init_strategy))
#
#        for trial_cfg in init_configs:
#            trial_cfg['federate.save_to'] = os.path.join(
#                global_cfg.hpo.working_folder,
#                "{}.pth".format(config2str(trial_cfg)))
#
#        return init_configs
#
#    def _stop_criterion(self, configs, last_results):
#        if last_results is not None:
#            if (global_cfg.hpo.larger_better
#                    and last_results.iloc[0]['performance'] >=
#                    global_cfg.hpo.pbt.perf_threshold) or (
#                        (not global_cfg.hpo.larger_better)
#                        and last_results.iloc[0]['performance'] <=
#                        global_cfg.hpo.pbt.perf_threshold):
#                return True
#        return self._stage >= global_cfg.hpo.pbt.max_stage
#
#    def _generate_next_population(self, configs, perfs):
#        next_generation = []
#        for i in range(len(configs)):
#            new_cfg = deepcopy(configs[i])
#            # exploit
#            j = np.random.randint(len(configs))
#            if i != j and (
#                (global_cfg.hpo.larger_better and perfs[j] > perfs[i]) or
#                ((not global_cfg.hpo.larger_better) and perfs[j] < perfs[i])):
#                new_cfg['federate.restore_from'] = configs[j][
#                    'federate.save_to']
#                # explore
#                for k in new_cfg:
#                    if isinstance(new_cfg[k], float):
#                        # according to the exploration strategy of the PBT
#                        paper
#                        new_cfg[k] *= float(np.random.choice([0.8, 1.2]))
#            else:
#                new_cfg['federate.restore_from'] = configs[i][
#                    'federate.save_to']
#
#            # update save path
#            tmp_cfg = dict()
#            for k in new_cfg:
#                if k in self._original_search_space:
#                    tmp_cfg[k] = new_cfg[k]
#            new_cfg['federate.save_to'] = os.path.join(
#                global_cfg.hpo.working_folder,
#                "{}.pth".format(config2str(tmp_cfg)))
#
#            next_generation.append(new_cfg)
#
#        return next_generation
