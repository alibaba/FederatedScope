# Implement TPE_MD, TPE_HB in `optuna`. from
# https://raw.githubusercontent.com/automl/HPOBenchExperimentUtils/master
# /HPOBenchExperimentUtils/optimizer/optuna_optimizer.py

import ConfigSpace as CS
import numpy as np
import time
import random
import optuna
import logging
from functools import partial
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial

from fedhpob.config import fhb_cfg
from fedhpob.utils.monitor import Monitor

logging.basicConfig(level=logging.WARNING)


def precompute_sh_iters(min_budget, max_budget, eta):
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    return max_SH_iter


def precompute_budgets(max_budget, eta, max_SH_iter):
    s0 = -np.linspace(start=max_SH_iter - 1, stop=0, num=max_SH_iter)
    budgets = max_budget * np.power(eta, s0)
    return budgets


def sample_config_from_optuna(trial: Trial, cs: CS.ConfigurationSpace):
    config = {}
    for hp_name in cs:
        hp = cs.get_hyperparameter(hp_name)

        if isinstance(hp, CS.UniformFloatHyperparameter):
            value = float(
                trial.suggest_float(name=hp_name,
                                    low=hp.lower,
                                    high=hp.upper,
                                    log=hp.log))

        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            value = int(
                trial.suggest_int(name=hp_name,
                                  low=hp.lower,
                                  high=hp.upper,
                                  log=hp.log))

        elif isinstance(hp, CS.CategoricalHyperparameter):
            hp_type = type(hp.default_value)
            value = hp_type(
                trial.suggest_categorical(name=hp_name, choices=hp.choices))

        elif isinstance(hp, CS.OrdinalHyperparameter):
            num_vars = len(hp.sequence)
            index = trial.suggest_int(hp_name,
                                      low=0,
                                      high=num_vars - 1,
                                      log=False)
            hp_type = type(hp.default_value)
            value = hp.sequence[index]
            value = hp_type(value)

        else:
            raise ValueError(
                f'Please implement the support for hps of type {type(hp)}')

        config[hp.name] = value
    return config


def run_optuna(cfg):
    def objective(trial, benchmark, valid_budgets, configspace):
        config = sample_config_from_optuna(trial, configspace)
        res = None
        for budget in valid_budgets:
            main_fidelity = {
                'round': int(budget),
                'sample_client': cfg.benchmark.sample_client
            }
            t_start = time.time()
            res = benchmark(config,
                            main_fidelity,
                            seed=random.randint(1, 99),
                            key='val_avg_loss',
                            fhb_cfg=cfg)
            monitor(res=res, sim_time=time.time() - t_start, budget=budget)
            trial.report(res['function_value'], step=budget)

            if trial.should_prune():
                raise optuna.TrialPruned()

        assert res is not None
        return res['function_value']

    monitor = Monitor(cfg)
    benchmark = cfg.benchmark.cls[0][cfg.benchmark.type](
        cfg.benchmark.model,
        cfg.benchmark.data,
        cfg.benchmark.algo,
        device=cfg.benchmark.device)
    sampler = TPESampler(seed=cfg.optimizer.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    if cfg.optimizer.type == 'tpe_md':
        pruner = MedianPruner()
        sh_iters = precompute_sh_iters(cfg.optimizer.min_budget,
                                       cfg.optimizer.max_budget,
                                       cfg.optimizer.optuna.reduction_factor)
        valid_budgets = precompute_budgets(
            cfg.optimizer.max_budget, cfg.optimizer.optuna.reduction_factor,
            sh_iters)
    elif cfg.optimizer.type == 'tpe_hb':
        pruner = HyperbandPruner(
            min_resource=cfg.optimizer.min_budget,
            max_resource=cfg.optimizer.max_budget,
            reduction_factor=cfg.optimizer.optuna.reduction_factor)
        pruner._try_initialization(study=None)
        valid_budgets = [
            cfg.optimizer.min_budget * cfg.optimizer.optuna.reduction_factor**i
            for i in range(pruner._n_brackets)
        ]
    else:
        raise NotImplementedError

    study.optimize(func=partial(
        objective,
        benchmark=benchmark,
        valid_budgets=valid_budgets,
        configspace=cfg.benchmark.configuration_space[0]),
                   timeout=None,
                   n_trials=cfg.optimizer.n_iterations)
    return monitor.history_results


if __name__ == "__main__":
    results = []
    for opt_name in ['tpe_md', 'tpe_hb']:
        fhb_cfg.optimizer.type = opt_name
        results.append(run_optuna(fhb_cfg))
    print(results)
