# Implement GS with `ConfigSpace.util.generate_grid`

import time
import random
import logging
import ConfigSpace as CS
from ConfigSpace.util import generate_grid

from fedhpob.config import fhb_cfg
from fedhpob.utils.monitor import Monitor

logging.basicConfig(level=logging.WARNING)


def run_grid_search(cfg):
    monitor = Monitor(cfg)
    cfg = cfg.clone()
    benchmark = cfg.benchmark.cls[0][cfg.benchmark.type](
        cfg.benchmark.model,
        cfg.benchmark.data,
        cfg.benchmark.algo,
        device=cfg.benchmark.device)
    # configspace = cfg.benchmark.configuration_space[0]
    # This is only for FEMNIST grid search exp
    new_configspace = CS.ConfigurationSpace()
    new_configspace.add_hyperparameter(
        CS.CategoricalHyperparameter('batch', choices=[16, 64]))
    new_configspace.add_hyperparameter(
        CS.CategoricalHyperparameter('dropout', choices=[0.0, 0.5]))
    new_configspace.add_hyperparameter(
        CS.CategoricalHyperparameter(
            'lr', choices=[0.01, 0.02783, 0.07743, 0.21544, 0.59948, 1.0]))
    new_configspace.add_hyperparameter(
        CS.CategoricalHyperparameter('step', choices=[2, 4]))
    new_configspace.add_hyperparameter(
        CS.CategoricalHyperparameter('wd', choices=[0.0, 0.1]))
    grid = generate_grid(new_configspace)
    print(len(grid))
    budget = cfg.optimizer.max_budget
    history = []
    # TODO: Add a gap here
    for idx in range(999999999):
        config = grid[idx % len(grid)]
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
        history.append(res)
        monitor(res=res, sim_time=time.time() - t_start, budget=budget)
    return history


if __name__ == "__main__":
    results = []
    for opt_name in ['gs']:
        fhb_cfg.optimizer.type = opt_name
        results.append(run_grid_search(fhb_cfg))
    print(results)
