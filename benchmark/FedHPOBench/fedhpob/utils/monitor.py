import os
import time
import json
import logging

import numpy as np

from fedhpob.utils.util import cfg2name

logging.basicConfig(level=logging.WARNING)


class Monitor(object):
    def __init__(self, cfg):
        self.limit_time = cfg.optimizer.limit_time
        self.last_timestamp = time.time()
        self.best_value = np.inf
        self.consumed_time, self.budget, self.cnt = 0, 0, 0
        self.logs = []
        self.cfg = cfg

    def __call__(self, res, sim_time=0, *args, **kwargs):
        self._check_and_log(res['cost'])
        # minus the time consumed in simulation and plus estimated time.
        self.consumed_time += (time.time() - self.last_timestamp - sim_time +
                               res['cost'])
        self.cnt += 1
        if res['function_value'] < self.best_value or kwargs[
                'budget'] > self.budget:
            self.budget = kwargs['budget']
            self.best_value = res['function_value']
        self.logs.append({
            'Try': self.cnt,
            "Consumed": self.consumed_time,
            'best_value': self.best_value,
            'cur_results': res
        })
        logging.warning(
            f'Try: {self.cnt}, Consumed: {self.consumed_time}, best_value:'
            f' {self.best_value}, cur_results: {res}')
        self.last_timestamp = time.time()

    def _check_and_log(self, cost):
        if self.consumed_time + cost > self.limit_time:
            # TODO: record time and cost
            logging.warning(
                f'Time has been consumed, no time for next try (cost: {cost})!'
            )
            out_file = cfg2name(self.cfg)
            with open(out_file, 'w') as f:
                for line in self.logs:
                    f.write(json.dumps(line) + "\n")
            os._exit(1)
