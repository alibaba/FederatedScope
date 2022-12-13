import numpy as np
import pandas as pd


class Node(object):
    def __init__(self,
                 status='on',
                 feature_idx=None,
                 feature_value=None,
                 weight=None,
                 grad=None,
                 hess=None,
                 label=None,
                 indicator=None):
        self.member = None
        self.status = status
        self.feature_idx = feature_idx
        self.value_idx = None
        self.feature_value = feature_value
        self.weight = weight
        self.grad = grad
        self.hess = hess
        self.label = label
        self.indicator = indicator


class Tree(object):
    def __init__(self, max_depth):
        self.tree = [Node() for _ in range(2**max_depth - 1)]
