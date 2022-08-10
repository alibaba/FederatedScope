import os
import time
import math
import logging

from os.path import join as osp
import numpy as np
import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from hpbandster.optimizers.iterations import SuccessiveHalving

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MyBOHB(BOHB):
    def __init__(self, working_folder, **kwargs):
        self.working_folder = working_folder
        super(MyBOHB, self).__init__(**kwargs)

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta**s)
        ns = [max(int(n0 * (self.eta**(-i))), 1) for i in range(s + 1)]
        if os.path.exists(self.working_folder):
            self.clear_cache()
        return (SuccessiveHalving(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-s - 1):],
            config_sampler=self.config_generator.get_config,
            **iteration_kwargs))

    def clear_cache(self):
        # Clear cached ckpt
        for name in os.listdir(self.working_folder):
            if name.endswith('_fedex.yaml') or name.endswith('.pth'):
                os.remove(osp(self.working_folder, name))


def eval_in_fs(cfg, config, budget):
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.worker_builder import \
        get_client_cls, get_server_cls
    from federatedscope.core.fed_runner import FedRunner
    from federatedscope.autotune.utils import config2cmdargs

    # Add FedEx related keys to config
    if 'hpo.table.idx' in config.keys():
        idx = config['hpo.table.idx']
        config['hpo.fedex.ss'] = osp(cfg.hpo.working_folder,
                                     f"{idx}_tmp_grid_search_space.yaml")
        config['federate.save_to'] = osp(cfg.hpo.working_folder,
                                         f"idx_{idx}.pth")
        config['federate.restore_from'] = osp(cfg.hpo.working_folder,
                                              f"idx_{idx}.pth")
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
    def __init__(self, cfg, ss, sleep_interval=0, *args, **kwargs):
        super(MyWorker, self).__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self.cfg = cfg
        self._ss = ss
        self._init_configs = []
        self._perfs = []

    def compute(self, config, budget, **kwargs):
        res = eval_in_fs(self.cfg, config, int(budget))
        self._init_configs.append(config)
        self._perfs.append(float(res))
        time.sleep(self.sleep_interval)
        return {'loss': float(res), 'info': res}

    def summarize(self):
        from federatedscope.autotune.utils import summarize_hpo_results
        results = summarize_hpo_results(self._init_configs,
                                        self._perfs,
                                        white_list=set(self._ss.keys()),
                                        desc=self.cfg.hpo.larger_better)
        logger.info(
            "========================== HPO Final ==========================")
        logger.info("\n{}".format(results))
        logger.info("====================================================")

        return results


def run_hpbandster(cfg, scheduler):
    config_space = scheduler._search_space
    if cfg.hpo.scheduler.startswith('wrap_'):
        ss = CS.ConfigurationSpace()
        ss.add_hyperparameter(config_space['hpo.table.idx'])
        config_space = ss
    NS = hpns.NameServer(run_id=cfg.hpo.scheduler, host='127.0.0.1', port=0)
    ns_host, ns_port = NS.start()
    w = MyWorker(sleep_interval=0,
                 ss=config_space,
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
        'eta': cfg.hpo.sha.elim_rate,
        'min_budget': cfg.hpo.sha.budgets[0],
        'max_budget': cfg.hpo.sha.budgets[-1],
        'working_folder': cfg.hpo.working_folder
    }
    optimizer = MyBOHB(**opt_kwargs)
    if cfg.hpo.sha.iter != 0:
        n_iterations = cfg.hpo.sha.iter
    else:
        n_iterations = -int(
            np.log(opt_kwargs['min_budget'] / opt_kwargs['max_budget']) /
            np.log(opt_kwargs['eta'])) + 1
    res = optimizer.run(n_iterations=n_iterations)
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
    all_runs = res.get_all_runs()
    w.summarize()

    return [x.info for x in all_runs]
