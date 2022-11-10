from itertools import product
from math import ceil
import numpy as np
import argparse



def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=0, type=int,
                        help='run in DEBUG mode if >0; sets number of clients and batches')

    # wrapper algorithm settings
    parser.add_argument('--rounds', default=800, type=int,
                        help='maximum number of communication rounds')
    parser.add_argument('--total', default=4000, type=int,
                        help='total number of communication rounds')
    parser.add_argument('--rate', default=2, type=int, help='elimination rate')
    parser.add_argument('--elim', default=4, type=int, help='number of elimination rounds')
    parser.add_argument('--eval', default=1, type=int, help='number of evaluation rounds')
    parser.add_argument('--discount', default=0.0, type=float,
                        help='discount factor for computing the validation score of an arm')

    # FedEx settings
    parser.add_argument('--batch', default=10, type=int, help='number of tasks per round')
    parser.add_argument('--configs', default=1, type=int,
                        help='''number of configs to optimize over with FedEx (use 1 for FedAvg):
                                - <-1: sample a random number between 1 and abs(args.configs)
                                - =-1: sample the number of arms given by the wrapper
                                - =0: sample a random number between 1 and the number of arms
                                - >0: sample the provided number, ignoring the number of arms''')
    parser.add_argument('--lr_only', action='store_true', help='tune only learning rate as a hyperparameter')
    parser.add_argument('--eps', default=0.0, type=float, help='multiplicative perturbation to client config, eps=0 is fedavg')
    parser.add_argument('--uniform', action='store_true',
                        help='run FedEx over a product set of single-parameter uniform grids')
    parser.add_argument('--random', action='store_true',
                        help='run FedEx over a product set of single-parameter random grids')
    parser.add_argument('--eta0', default=0.0, type=float,
                        help='FedEx initial step size; if 0.0 uses FedEx default')
    parser.add_argument('--sched', default='aggressive', type=str, help='FedEx step size sched')
    parser.add_argument('--cutoff', default=0.0, type=float,
                        help='stop updating FedEx config distribution if entropy below this cutoff')
    parser.add_argument('--baseline', default=-1.0, type=float,
                        help='''how FedEx computes the baseline:
                                - >=-1.0,<0.0: sample discount factor from [0.0, abs(args.baseline))
                                - =0.0: use the most recent value
                                - >0.0,<1.0: use geometrically discounted mean with this factor
                                - =1.0: use the mean of all values''')
    parser.add_argument('--diff', action='store_true',
                        help='use difference between refine and global as FedEx objective')
    parser.add_argument('--stop', action='store_true',
                        help='stop updating FedEx config distribution after last elimination')

    # evaluation settings
    parser.add_argument('--mle', action='store_true', help='use MLE config at test time')
    parser.add_argument('--loss', action='store_true', help='use loss instead of error')
    parser.add_argument('--eval_global', action='store_true', help='use global error as elimination metric instead of refine')

# data settings
    parser.add_argument('--val', default=0.2, type=float, help='proportion of training set to use for validation')
    parser.add_argument('--num-clients', default=500, type=int, help='number of clients')


    return parser.parse_args()


def random_search(max_resources=500, total_resources=2000):
    '''returns a random search rate and schedule for use by 'successive_elimination'
    Args:
        max_resources: most resources (steps) assigned to single arm
        total_resources: overall resource limit
    Returns:
        elimination rate as an int, elimination schedule as a list
    '''

    assert max_resources > 0, "max_resources must be positive"
    return int(total_resources / max_resources), [max_resources]

def get_schedule(
                 max_resources=500,
                 total_resources=2000,
                 elim_rate=3,
                 num_elim=0,
                 num_eval=1,
                 ):
    '''returns rate and schedule for use by 'successive_elimination'
    Args:
        max_resources: most resources (steps) assigned to single arm
        total_resources: overall resource limit
        elim_rate: multiplicative elimination rate
        num_elim: number of elimination rounds; if 0 runs random search
        num_eval: number of evaluation rounds
    Returns:
        elimination rate as an int, elimination schedule as a list, evaluation schedule as a list
    '''

    assert max_resources <= total_resources, "max_resources cannot be greater than total_resources"
    assert elim_rate > 1, "elim_rate must be greater than 1"
    assert num_eval <= total_resources, "num_eval cannot be greater than total_resources"
    if num_elim:
        diff = total_resources - max_resources
        geos = (elim_rate**(num_elim+1) - 1) / (elim_rate-1)
        u = int(diff / (geos-num_elim-1))
        resources = 0
        v = lambda i: 1 + ceil((diff+(num_elim-geos+elim_rate**i)*u) / (elim_rate**i-1))
        for opt in product(*(range(u, v(i)) for i in reversed(range(1, num_elim+1)))):
            used = max_resources + sum((elim_rate**i-1)*r
                                       for i, r in zip(reversed(range(1, num_elim+1)), opt))
            if resources <= used <= total_resources:
                best, resources = opt, used
        assert not 0 in best, "invalid: use more resources or fewer eliminations, or increase rate"
        elim_sched = list(np.cumsum(best)) + [max_resources]
    else:
        print('random search')
        elim_rate, elim_sched = random_search(max_resources=max_resources,
                                              total_resources=total_resources)

    eval_sched = [int(step) for step in np.linspace(0, total_resources, num_eval+1)[1:]]
    return elim_rate, elim_sched, eval_sched



args = parse()

args.total = 3000
args.elim = 3
args.rate = 3
args.rounds = 500

elim_rate, elim_sched, eval_sched = get_schedule(
    max_resources=args.rounds,
    total_resources=args.total,
    elim_rate=args.rate,
    num_elim=args.elim,
    num_eval=args.eval,
)
print(
    'max_resources=',args.rounds,
    '  total_resources=',args.total,
    '  elim_rate=',args.rate,
    '  num_elim=',args.elim,
    '  num_eval=',args.eval,
)
print(elim_rate)
ss = np.array(elim_sched[1:]) - np.array(elim_sched[:-1])
print('schedule: ', ss)
a = 0
ss = ss[::-1]
for i in range(len(ss)):
    a += args.rate ** (i+1) * ss[i]
print(a)

# import os
# import pdb
# from federatedscope.core.auxiliaries.utils import logfile_2_wandb_dict
#
# with open(os.path.join('exp/pfedhpo_cifar/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.1_lstep1_/', "eval_results.log"),
#           "r") as exp_log_f:
#     # track the prediction related performance
#     all_log_res, exp_stop_normal, last_line, log_res_best = logfile_2_wandb_dict(exp_log_f, raw_out=False)
#
# pdb.set_trace()