import numpy as np


class Node(object):
    def __init__(self,
                 status='on',
                 feature_idx=None,
                 feature_value=None,
                 weight=None,
                 grad=None,
                 hess=None,
                 indicator=None):
        self.member = None
        self.status = status
        self.feature_idx = feature_idx
        self.value_idx = None
        self.feature_value = feature_value
        self.weight = weight
        self.grad = grad
        self.hess = hess
        self.indicator = indicator


class Tree(object):
    def __init__(self, max_depth, lambda_, gamma):
        self.tree = [Node() for _ in range(2**max_depth - 1)]
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth

    def __getitem__(self, item):
        return self.tree[item]

    def _gain(self, grad, hess):
        return np.power(grad, 2) / (hess + self.lambda_)

    def cal_gain(self, grad, hess, split_idx):
        left_grad = np.sum(grad[:split_idx])
        right_grad = np.sum(grad[split_idx:])
        left_hess = np.sum(hess[:split_idx])
        right_hess = np.sum(hess[split_idx:])
        left_gain = self._gain(left_grad, left_hess)
        right_gain = self._gain(right_grad, right_hess)
        total_gain = self._gain(left_grad + right_grad, left_hess + right_hess)
        return (left_gain + right_gain - total_gain) * 0.5 - self.gamma

    def split_childern(self, data, feature_value):
        left_index = [1 if x < feature_value else 0 for x in data]
        right_index = [1 if x >= feature_value else 0 for x in data]
        return left_index, right_index

    def set_weight(self, node_num):
        sum_of_g = np.sum(self.tree[node_num].grad)
        sum_of_h = np.sum(self.tree[node_num].hess)
        weight = -sum_of_g / (sum_of_h + self.lambda_)
        self.tree[node_num].weight = weight

    def set_status(self, node_num, status='off'):
        self.tree[node_num].status = status

    def update_child(self, node_num, left_child, right_child):
        self.tree[2 * node_num +
                  1].grad = self.tree[node_num].grad * left_child
        self.tree[2 * node_num +
                  1].hess = self.tree[node_num].hess * left_child
        self.tree[2 * node_num +
                  1].indicator = self.tree[node_num].indicator * left_child
        self.tree[2 * node_num +
                  2].grad = self.tree[node_num].grad * right_child
        self.tree[2 * node_num +
                  2].hess = self.tree[node_num].hess * right_child
        self.tree[2 * node_num +
                  2].indicator = self.tree[node_num].indicator * right_child


class MultipleTrees(object):
    def __init__(self, max_depth, lambda_, gamma, num_of_trees):
        self.trees = [
            Tree(max_depth=max_depth, lambda_=lambda_, gamma=gamma)
            for _ in range(num_of_trees)
        ]
        self.num_of_trees = num_of_trees
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth

    def __getitem__(self, item):
        return self.trees[item]
