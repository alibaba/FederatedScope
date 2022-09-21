# Implement RS, BO_KDE, HB, BOHB in `hpbandster`.

import time
import random
import logging
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB, HyperBand, RandomSearch

from fedhpob.config import fhb_cfg
from fedhpob.utils.monitor import Monitor

logging.basicConfig(level=logging.WARNING)


class MyWorker(Worker):
    def __init__(self,
                 benchmark,
                 monitor,
                 sleep_interval=0,
                 cfg=None,
                 **kwargs):
        super(MyWorker, self).__init__(**kwargs)
        self.benchmark = benchmark
        self.monitor = monitor
        self.sleep_interval = sleep_interval
        self.cfg = cfg

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the
        budget) For dramatization, the function can sleep for a given
        interval to emphasizes the speed ups achievable with parallel workers.
        Args:
            config: dictionary containing the sampled configurations by the
            optimizer
            budget: (float) amount of time/epochs/etc. the model can use to
            train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        main_fidelity = {
            'round': int(budget),
            'sample_client': self.cfg.benchmark.sample_client
        }
        t_start = time.time()
        res = self.benchmark(config,
                             main_fidelity,
                             seed=random.randint(1, 99),
                             key='val_avg_loss',
                             fhb_cfg=self.cfg)
        time.sleep(self.sleep_interval)
        self.monitor(res=res, sim_time=time.time() - t_start, budget=budget)
        return ({
            'loss': float(res['function_value']
                          ),  # this is a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information -
            # also mandatory
        })


def run_hpbandster(cfg):
    if cfg.optimizer.type == 'bo_kde':
        cfg.optimizer.min_budget = cfg.optimizer.max_budget
    monitor = Monitor(cfg)
    NS = hpns.NameServer(run_id=cfg.optimizer.type, host='127.0.0.1', port=0)
    ns_host, ns_port = NS.start()
    cfg = cfg.clone()
    benchmark = cfg.benchmark.cls[0][cfg.benchmark.type](
        cfg.benchmark.model,
        cfg.benchmark.data,
        cfg.benchmark.algo,
        device=cfg.benchmark.device)
    w = MyWorker(
        benchmark=benchmark,
        monitor=monitor,
        sleep_interval=0,
        cfg=cfg,
        nameserver='127.0.0.1',
        # nameserver=ns_host,
        nameserver_port=ns_port,
        run_id=cfg.optimizer.type)
    w.run(background=True)

    # Allow at most max_stages stages
    tmp = cfg.optimizer.max_budget
    for i in range(cfg.optimizer.hpbandster.max_stages):
        tmp /= cfg.optimizer.hpbandster.eta
    if tmp > cfg.optimizer.min_budget:
        cfg.optimizer.min_budget = tmp

    opt_kwargs = {
        'configspace': cfg.benchmark.configuration_space[0],
        'run_id': cfg.optimizer.type,
        'nameserver': '127.0.0.1',
        'nameserver_port': ns_port,
        'eta': cfg.optimizer.hpbandster.eta,
        'min_budget': cfg.optimizer.min_budget,
        'max_budget': cfg.optimizer.max_budget
    }
    if cfg.optimizer.type == 'rs':
        optimizer = RandomSearch(**opt_kwargs)
    elif cfg.optimizer.type == 'bo_kde':
        optimizer = BOHB(**opt_kwargs)
    elif cfg.optimizer.type == 'hb':
        optimizer = HyperBand(**opt_kwargs)
    elif cfg.optimizer.type == 'bohb':
        optimizer = BOHB(**opt_kwargs)
    else:
        raise NotImplementedError
    res = optimizer.run(n_iterations=cfg.optimizer.n_iterations)

    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
    all_runs = res.get_all_runs()
    return [x.info for x in all_runs]


if __name__ == "__main__":
    results = []
    for opt_name in ['rs', 'bo_kde', 'hb', 'bohb']:
        fhb_cfg.optimizer.type = opt_name
        results.append(run_hpbandster(fhb_cfg))
    print(results)
