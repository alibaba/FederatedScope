# import os
# import sys
# file_dir = os.path.join(os.path.dirname(__file__), '../..')
# sys.path.append(file_dir)
import logging
import math
import yaml

import numpy as np

from federatedscope.core.configs.config import global_cfg

logger = logging.getLogger(__name__)


def discretize(contd_choices, num_bkt):
    '''Discretize a given continuous search space into the given number of buckets.

    Arguments:
        contd_choices (Continuous): continuous choices.
        num_bkt (int): number of buckets.
    :returns: discritized choices.
    :rtype: Discrete
    '''
    if contd_choices[0] >= .0 and global_cfg.hpo.log_scale:
        loglb, logub = math.log(
            np.clip(contd_choices[0], 1e-8,
                    contd_choices[1])), math.log(contd_choices[1])
        if num_bkt == 1:
            choices = [math.exp(loglb + 0.5 * (logub - loglb))]
        else:
            bkt_size = (logub - loglb) / (num_bkt - 1)
            choices = [math.exp(loglb + i * bkt_size) for i in range(num_bkt)]
    else:
        if num_bkt == 1:
            choices = [
                contd_choices[0] + 0.5 * (contd_choices[1] - contd_choices[0])
            ]
        else:
            bkt_size = (contd_choices[1] - contd_choices[0]) / (num_bkt - 1)
            choices = [contd_choices[0] + i * bkt_size for i in range(num_bkt)]
    disc_choices = Discrete(*choices)
    return disc_choices


class Continuous(tuple):
    """Represents a continuous search space, e.g., in the range [0.001, 0.1].
    """
    def __new__(cls, lb, ub):
        assert ub >= lb, "Invalid configuration where ub:{} is less than " \
                         "lb:{}".format(ub, lb)
        return tuple.__new__(cls, [lb, ub])

    def __repr__(self):
        return "Continuous(%s,%s)" % self

    def sample(self):
        """Sample a value from this search space.

        :returns: the sampled value.
        :rtype: float
        """
        if self[0] >= .0 and global_cfg.hpo.log_scale:
            loglb, logub = math.log(np.clip(self[0], 1e-8,
                                            self[1])), math.log(self[1])
            return math.exp(loglb + np.random.rand() * (logub - loglb))
        else:
            return float(self[0] + np.random.rand() * (self[1] - self[0]))

    def grid(self, grid_cnt):
        """Generate a given nunber of grids from this search space.

        Arguments:
            grid_cnt (int): the number of grids.
        :returns: the sampled value.
        :rtype: float
        """
        discretized = discretize(self, grid_cnt)
        return list(discretized)


def contd_constructor(loader, node):
    value = loader.construct_scalar(node)
    lb, ub = map(float, value.split(','))
    return Continuous(lb, ub)


yaml.add_constructor(u'!contd', contd_constructor)


class Discrete(tuple):
    """Represents a discrete search space, e.g., {'abc', 'ijk', 'xyz'}.
    """
    def __new__(cls, *args):
        return tuple.__new__(cls, args)

    def __repr__(self):
        return "Discrete(%s)" % ','.join(map(str, self))

    def sample(self):
        """Sample a value from this search space.

        :returns: the sampled value.
        :rtype: depends on the original choices.
        """

        return self[np.random.randint(len(self))]

    def grid(self, grid_cnt):
        num_original = len(self)
        assert grid_cnt <= num_original, "There are only {} choices to " \
                                         "produce grids, but {} " \
                                         "required".format(num_original,
                                                           grid_cnt)
        if grid_cnt == 1:
            selected = [self[len(self) // 2]]
        else:
            optimistic_step_size = (num_original - 1) // grid_cnt
            between_end_len = optimistic_step_size * (grid_cnt - 1)
            remainder = (num_original - 1) - between_end_len
            one_side_remainder = remainder // 2 if remainder % 2 == 0 else \
                remainder // 2 + 1
            if one_side_remainder <= optimistic_step_size // 2:
                step_size = optimistic_step_size
            else:
                step_size = (num_original - 1) // (grid_cnt - 1)
            covered_range = (grid_cnt - 1) * step_size
            start_idx = (max(num_original - 1, 1) - covered_range) // 2
            selected = [
                self[j] for j in range(
                    start_idx,
                    min(start_idx +
                        grid_cnt * step_size, num_original), step_size)
            ]
        return selected


def disc_constructor(loader, node):
    value = loader.construct_sequence(node)
    return Discrete(*value)


yaml.add_constructor(u'!disc', disc_constructor)

# if __name__=="__main__":
#    obj = Continuous(0.0, 0.01)
#    print(obj.grid(1), obj.grid(2), obj.grid(3))
#    for _ in range(3):
#        print(obj.sample())
#    cfg.merge_from_list(['hpo.log_scale', 'True'])
#    print(obj.grid(1), obj.grid(2), obj.grid(3))
#    for _ in range(3):
#        print(obj.sample())
#
#    obj = Discrete('a', 'b', 'c')
#    print(obj.grid(1), obj.grid(2), obj.grid(3))
#    for _ in range(3):
#        print(obj.sample())
#    obj = Discrete(1, 2, 3, 4, 5)
#    print(obj.grid(1), obj.grid(2), obj.grid(3), obj.grid(4), obj.grid(5))
#    for _ in range(3):
#        print(obj.sample())
