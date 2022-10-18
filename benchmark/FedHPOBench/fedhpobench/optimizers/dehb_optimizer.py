"""
https://github.com/automl/DEHB/blob/master/examples/00_interfacing_DEHB.ipynb
How to use the DEHB Optimizer
1) Download the Source Code
git clone https://github.com/automl/DEHB.git
# We are currently using the first version of it.
cd DEHB
git checkout b8dcba7b38bf6e7fc8ce3e84ea567b66132e0eb5
2) Add the project to your Python Path
export PYTHONPATH=~/DEHB:$PYTHONPATH
3) Requirements
- dask distributed:
```
conda install dask distributed -c conda-forge
```
OR
```
python -m pip install dask distributed --upgrade
```
- Other things to install:
```
pip install numpy, ConfigSpace
```
"""

import time
import random
import logging
from dehb.optimizers import DE, DEHB
from fedhpob.config import fhb_cfg
from fedhpob.utils.monitor import Monitor

logging.basicConfig(level=logging.WARNING)


def run_dehb(cfg):
    def objective(config, budget=None):
        if cfg.optimizer.type == 'de':
            budget = cfg.optimizer.max_budget
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
        fitness, cost = res['function_value'], res['cost']
        return fitness, cost

    monitor = Monitor(cfg)
    benchmark = cfg.benchmark.cls[0][cfg.benchmark.type](
        cfg.benchmark.model,
        cfg.benchmark.data,
        cfg.benchmark.algo,
        device=cfg.benchmark.device)
    if cfg.optimizer.type == 'de':
        optimizer = DE(
            cs=cfg.benchmark.configuration_space[0],
            dimensions=len(
                cfg.benchmark.configuration_space[0].get_hyperparameters()),
            f=objective,
            pop_size=cfg.optimizer.dehb.de.pop_size,
            mutation_factor=cfg.optimizer.dehb.mutation_factor,
            crossover_prob=cfg.optimizer.dehb.crossover_prob,
            strategy=cfg.optimizer.dehb.strategy)
        traj, runtime, history = optimizer.run(
            generations=cfg.optimizer.n_iterations, verbose=False)
    elif cfg.optimizer.type == 'dehb':
        optimizer = DEHB(
            cs=cfg.benchmark.configuration_space[0],
            dimensions=len(
                cfg.benchmark.configuration_space[0].get_hyperparameters()),
            f=objective,
            strategy=cfg.optimizer.dehb.strategy,
            mutation_factor=cfg.optimizer.dehb.mutation_factor,
            crossover_prob=cfg.optimizer.dehb.crossover_prob,
            eta=cfg.optimizer.dehb.dehb.eta,
            min_budget=cfg.optimizer.min_budget,
            max_budget=cfg.optimizer.max_budget,
            generations=cfg.optimizer.dehb.dehb.gens,
            n_workers=1)
        traj, runtime, history = optimizer.run(
            iterations=cfg.optimizer.n_iterations, verbose=False)
    else:
        raise NotImplementedError

    return monitor.history_results


if __name__ == "__main__":
    # Please specific args for the experiment.
    results = []
    for opt_name in ['de', 'dehb']:
        fhb_cfg.optimizer.type = opt_name
        results.append(run_dehb(fhb_cfg))
    print(results)
