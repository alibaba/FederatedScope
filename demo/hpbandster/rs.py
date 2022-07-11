#import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers.randomsearch import RandomSearch
#from hpbandster.examples.commons import MyWorker

parser = argparse.ArgumentParser(
    description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',
                    type=float,
                    help='Minimum budget used during the optimization.',
                    default=1)
parser.add_argument('--max_budget',
                    type=float,
                    help='Maximum budget used during the optimization.',
                    default=27)
parser.add_argument('--n_iterations',
                    type=int,
                    help='Number of iterations performed by the optimizer',
                    default=4)
args = parser.parse_args()


def eval_fl_algo(x, b):
    from federatedscope.core.cmd_args import parse_args
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
    from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.fed_runner import FedRunner

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(
        "federatedscope/example_configs/single_process.yaml")
    # specify the configuration of interest
    init_cfg.merge_from_list([
        "train.optimizer.lr",
        float(x['lr']), "train.optimizer.weight_decay",
        float(x['wd']), "model.dropout",
        float(x["dropout"])
    ])
    # specify the budget
    init_cfg.merge_from_list(
        ["federate.total_round_num",
         int(b), "eval.freq",
         int(b)])

    update_logger(init_cfg, True)
    setup_seed(init_cfg.seed)

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global cfg object
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone())
    results = runner.run()

    # so that we could modify cfg in the next trial
    init_cfg.defrost()

    return results['client_summarized_weighted_avg']['test_avg_loss']


class MyWorker(Worker):
    def __init__(self, *args, sleep_interval=0, **kwargs):
        super(MyWorker, self).__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        #res = numpy.clip(config['x'] + numpy.random.randn()/budget, config['x']/2, 1.5*config['x'])
        res = eval_fl_algo(config, budget)
        time.sleep(self.sleep_interval)

        return ({
            'loss': float(
                res),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('lr',
                                          lower=1e-4,
                                          upper=1.0,
                                          log=True))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter('dropout', lower=.0, upper=.5))
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter('wd', choices=[0.0, 0.5]))
        return config_space


def main():
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    w = MyWorker(sleep_interval=0, nameserver='127.0.0.1', run_id='example1')
    w.run(background=True)

    #bohb = BOHB(  configspace = w.get_configspace(),
    #          run_id = 'example1', nameserver='127.0.0.1',
    #          min_budget=args.min_budget, max_budget=args.max_budget
    #       )
    rs = RandomSearch(configspace=w.get_configspace(),
                      run_id='example1',
                      nameserver='127.0.0.1',
                      min_budget=args.min_budget,
                      max_budget=args.max_budget)
    #res = bohb.run(n_iterations=args.n_iterations)
    res = rs.run(n_iterations=args.n_iterations)

    #bohb.shutdown(shutdown_workers=True)
    rs.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' %
          len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' %
          (sum([r.budget for r in res.get_all_runs()]) / args.max_budget))


if __name__ == "__main__":
    main()
