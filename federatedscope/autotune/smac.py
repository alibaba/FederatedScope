import logging
import numpy as np
import ConfigSpace as CS
from federatedscope.autotune.utils import eval_in_fs
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_smac(cfg, scheduler, client_cfgs=None):
    init_configs = []
    perfs = []

    def optimization_function_wrapper(config):
        budget = cfg.hpo.sha.budgets[-1]
        res = eval_in_fs(cfg, config, budget, client_cfgs)
        config = dict(config)
        config['federate.total_round_num'] = budget
        init_configs.append(config)
        perfs.append(res)
        logger.info(f'Evaluate the {len(perfs)-1}-th config '
                    f'{config}, and get performance {res}')
        return res

    def summarize():
        from federatedscope.autotune.utils import summarize_hpo_results
        results = summarize_hpo_results(init_configs,
                                        perfs,
                                        white_list=set(config_space.keys()),
                                        desc=cfg.hpo.larger_better)
        logger.info(
            "========================== HPO Final ==========================")
        logger.info("\n{}".format(results))
        logger.info("====================================================")

        return perfs

    config_space = scheduler._search_space
    if cfg.hpo.scheduler.startswith('wrap_'):
        ss = CS.ConfigurationSpace()
        ss.add_hyperparameter(config_space['hpo.table.idx'])
        config_space = ss

    if cfg.hpo.sha.iter != 0:
        n_iterations = cfg.hpo.sha.iter
    else:
        n_iterations = -int(
            np.log(cfg.hpo.sha.budgets[0] / cfg.hpo.sha.budgets[-1]) /
            np.log(cfg.hpo.sha.elim_rate)) + 1

    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": n_iterations,
        "cs": config_space,
        "output_dir": cfg.hpo.working_folder,
        "deterministic": "true",
        "limit_resources": False
    })

    if cfg.hpo.scheduler.endswith('bo_gp'):
        smac = SMAC4BB(model_type='gp',
                       scenario=scenario,
                       tae_runner=optimization_function_wrapper)
    elif cfg.hpo.scheduler.endswith('bo_rf'):
        smac = SMAC4HPO(scenario=scenario,
                        tae_runner=optimization_function_wrapper)
    else:
        raise NotImplementedError
    try:
        smac.optimize()
    finally:
        smac.solver.incumbent
    summarize()
    return perfs
