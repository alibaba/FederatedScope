import time
import math
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB


def eval_in_FS(x, b):
    return 0


class MyWorker(Worker):
    def __init__(self, sleep_interval=0, cfg=None, *args, **kwargs):
        super(MyWorker, self).__init__(**kwargs)
        self.sleep_interval = sleep_interval
        self.cfg = cfg

    def compute(self, config, budget, **kwargs):
        # TODO: implement this
        res = eval_in_FS(config, budget)
        time.sleep(self.sleep_interval)
        return ({'loss': float(res), 'info': res})


def run_hpbandster(cfg, scheduler):
    if cfg.hpo.scheduler in ['bo_kde', 'bohb']:
        config_space = scheduler._search_space
    elif cfg.hpo.scheduler in ['wrap_bo_kde', 'wrap_bohb']:
        config_space = scheduler._init_configs

    NS = hpns.NameServer(run_id=cfg.hpo.scheduler, host='127.0.0.1', port=0)
    ns_host, ns_port = NS.start()
    w = MyWorker(sleep_interval=0,
                 cfg=cfg,
                 nameserver='127.0.0.1',
                 nameserver_port=ns_port,
                 run_id=cfg.optimizer.type)
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
    res = optimizer.run(n_iterations=cfg.optimizer.n_iterations)

    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
    all_runs = res.get_all_runs()

    return [x.info for x in all_runs]
