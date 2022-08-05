# Implement BO_GP, BO_RF in `smac`.

import time
import random
import logging
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from fedhpob.config import fhb_cfg
from fedhpob.utils.monitor import Monitor

logging.basicConfig(level=logging.WARNING)


def run_smac(cfg):
    def optimization_function_wrapper(config):
        """ Helper-function: simple wrapper to use the benchmark with smac"""
        budget = int(cfg.optimizer.max_budget)
        main_fidelity = {
            'round': budget,
            'sample_client': cfg.benchmark.sample_client
        }
        t_start = time.time()
        res = benchmark(config,
                        main_fidelity,
                        seed=random.randint(1, 99),
                        key='val_avg_loss',
                        fhb_cfg=cfg)
        monitor(res=res, sim_time=time.time() - t_start, budget=budget)
        return res['function_value']

    monitor = Monitor(cfg)
    benchmark = cfg.benchmark.cls[0][cfg.benchmark.type](
        cfg.benchmark.model,
        cfg.benchmark.data,
        cfg.benchmark.algo,
        device=cfg.benchmark.device)

    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": cfg.optimizer.
        n_iterations,  # Max number of function evaluations
        "cs": cfg.benchmark.configuration_space[0],
        "output_dir": cfg.benchmark.type,
        "deterministic": "true",
        "limit_resources": False
    })
    if cfg.optimizer.type == 'bo_gp':
        smac = SMAC4BB(model_type='gp',
                       scenario=scenario,
                       tae_runner=optimization_function_wrapper)
    elif cfg.optimizer.type == 'bo_rf':
        smac = SMAC4HPO(scenario=scenario,
                        tae_runner=optimization_function_wrapper)
    else:
        raise NotImplementedError

    try:
        smac.optimize()
    finally:
        smac.solver.incumbent

    return monitor.history_results


if __name__ == "__main__":
    results = []
    for opt_name in ['bo_gp', 'bo_rf']:
        fhb_cfg.optimizer.type = opt_name
        results.append(run_smac(fhb_cfg))
    print(results)
