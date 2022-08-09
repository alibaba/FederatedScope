import time
import math
import logging

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns

from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

logging.basicConfig(level=logging.WARNING)


def eval_in_fs(cfg, config, budget):
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.worker_builder import \
        get_client_cls, get_server_cls
    from federatedscope.core.fed_runner import FedRunner
    from federatedscope.autotune.utils import config2cmdargs

    # Global cfg
    trial_cfg = cfg.clone()
    # specify the configuration of interest
    trial_cfg.merge_from_list(config2cmdargs(config))
    # specify the budget
    trial_cfg.merge_from_list(
        ["federate.total_round_num",
         int(budget), "eval.freq",
         int(budget)])
    setup_seed(trial_cfg.seed)
    data, modified_config = get_data(config=trial_cfg.clone())
    trial_cfg.merge_from_other_cfg(modified_config)
    trial_cfg.freeze()
    Fed_runner = FedRunner(data=data,
                           server_class=get_server_cls(trial_cfg),
                           client_class=get_client_cls(trial_cfg),
                           config=trial_cfg.clone())
    results = Fed_runner.run()
    key1, key2 = trial_cfg.hpo.metric.split('.')
    return results[key1][key2]


class MyWorker(Worker):
    def __init__(self, cfg, sleep_interval=0, *args, **kwargs):
        super(MyWorker, self).__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self.cfg = cfg

    def compute(self, config, budget, **kwargs):
        res = eval_in_fs(self.cfg, config, int(budget))
        time.sleep(self.sleep_interval)
        return {'loss': float(res), 'info': res}


def run_hpbandster(cfg, scheduler):
    if cfg.hpo.scheduler in ['bo_kde', 'bohb']:
        config_space = scheduler._search_space
    elif cfg.hpo.scheduler in ['wrap_bo_kde', 'wrap_bohb']:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(
            CSH.CategoricalHyperparameter('cfg',
                                          choices=scheduler._init_configs))

    NS = hpns.NameServer(run_id=cfg.hpo.scheduler, host='127.0.0.1', port=0)
    ns_host, ns_port = NS.start()
    w = MyWorker(sleep_interval=0,
                 cfg=cfg,
                 nameserver='127.0.0.1',
                 nameserver_port=ns_port,
                 run_id=cfg.hpo.scheduler)
    w.run(background=True)
    opt_kwargs = {
        'configspace': config_space,
        'run_id': cfg.hpo.scheduler,
        'nameserver': '127.0.0.1',
        'nameserver_port': ns_port,
        'eta': math.log(cfg.hpo.sha.budgets[-1] / cfg.hpo.sha.budgets[0],
                        len(cfg.hpo.sha.budgets)),
        'min_budget': cfg.hpo.sha.budgets[0],
        'max_budget': cfg.hpo.sha.budgets[-1]
    }
    optimizer = BOHB(**opt_kwargs)
    res = optimizer.run(n_iterations=4)

    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
    all_runs = res.get_all_runs()

    return [x.info for x in all_runs]
