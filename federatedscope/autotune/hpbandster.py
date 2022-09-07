import os
import time
import logging

from os.path import join as osp
import numpy as np
import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB, HyperBand, RandomSearch

from federatedscope.autotune.utils import eval_in_fs

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def clear_cache(working_folder):
    # Clear cached ckpt
    for name in os.listdir(working_folder):
        if name.endswith('.pth'):
            os.remove(osp(working_folder, name))


class MyRandomSearch(RandomSearch):
    def __init__(self, working_folder, **kwargs):
        self.working_folder = working_folder
        super(MyRandomSearch, self).__init__(**kwargs)


class MyBOHB(BOHB):
    def __init__(self, working_folder, **kwargs):
        self.working_folder = working_folder
        super(MyBOHB, self).__init__(**kwargs)

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        if os.path.exists(self.working_folder):
            clear_cache(self.working_folder)
        return super(MyBOHB, self).get_next_iteration(iteration,
                                                      iteration_kwargs)


class MyHyperBand(HyperBand):
    def __init__(self, working_folder, **kwargs):
        self.working_folder = working_folder
        super(MyHyperBand, self).__init__(**kwargs)

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        if os.path.exists(self.working_folder):
            clear_cache(self.working_folder)
        return super(MyHyperBand,
                     self).get_next_iteration(iteration, iteration_kwargs)


class MyWorker(Worker):
    def __init__(self,
                 cfg,
                 ss,
                 sleep_interval=0,
                 client_cfgs=None,
                 *args,
                 **kwargs):
        super(MyWorker, self).__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self.cfg = cfg
        self.client_cfgs = client_cfgs
        self._ss = ss
        self._init_configs = []
        self._perfs = []

    def compute(self, config, budget, **kwargs):
        res = eval_in_fs(self.cfg, config, int(budget), self.client_cfgs)
        config = dict(config)
        config['federate.total_round_num'] = budget
        self._init_configs.append(config)
        self._perfs.append(float(res))
        time.sleep(self.sleep_interval)
        logger.info(f'Evaluate the {len(self._perfs)-1}-th config '
                    f'{config}, and get performance {res}')
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


def run_hpbandster(cfg, scheduler, client_cfgs=None):
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
                 run_id=cfg.hpo.scheduler,
                 client_cfgs=client_cfgs)
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
    if cfg.hpo.scheduler in ['rs', 'wrap_rs']:
        optimizer = MyRandomSearch(**opt_kwargs)
    elif cfg.hpo.scheduler in ['hb', 'wrap_hb']:
        optimizer = MyHyperBand(**opt_kwargs)
    elif cfg.hpo.scheduler in ['bo_kde', 'bohb', 'wrap_bo_kde', 'wrap_bohb']:
        optimizer = MyBOHB(**opt_kwargs)
    else:
        raise ValueError

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
