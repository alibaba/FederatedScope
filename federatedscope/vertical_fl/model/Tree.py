import numpy as np


class Node(object):
    def __init__(self,
                 status='on',
                 feature_idx=None,
                 feature_value=None,
                 weight=None,
                 grad=None,
                 hess=None,
                 indicator=None,
                 label=None):
        self.member = None
        self.status = status
        self.feature_idx = feature_idx
        self.value_idx = None
        self.feature_value = feature_value
        self.weight = weight
        self.grad = grad
        self.hess = hess
        self.indicator = indicator
        self.label = label


class Tree(object):
    def __init__(self, max_depth, lambda_, gamma):
        self.tree = [Node() for _ in range(2**max_depth - 1)]
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth

    def __getitem__(self, item):
        return self.tree[item]

    def split_childern(self, data, feature_value):
        left_index = [1 if x < feature_value else 0 for x in data]
        right_index = [1 if x >= feature_value else 0 for x in data]
        return left_index, right_index

    def set_status(self, node_num, status='off'):
        self.tree[node_num].status = status

    def check_empty_child(self, node_num, split_idx, order):
        indicator = self.tree[node_num].indicator[order]
        if np.sum(indicator[:split_idx]) == 0 or np.sum(
                indicator[split_idx:]) == 0:
            return True

        return False


class XGBTree(Tree):
    def __init__(self, max_depth, lambda_, gamma):
        super().__init__(max_depth, lambda_, gamma)

    def _gain(self, grad, hess):
        return np.power(grad, 2) / (hess + self.lambda_)

    def cal_gain(self, grad, hess, split_idx, node_num):
        left_grad = np.sum(grad[:split_idx])
        right_grad = np.sum(grad[split_idx:])
        left_hess = np.sum(hess[:split_idx])
        right_hess = np.sum(hess[split_idx:])
        left_gain = self._gain(left_grad, left_hess)
        right_gain = self._gain(right_grad, right_hess)
        total_gain = self._gain(left_grad + right_grad, left_hess + right_hess)

        return (left_gain + right_gain - total_gain) * 0.5 - self.gamma

    def set_weight(self, node_num):
        sum_of_g = np.sum(self.tree[node_num].grad)
        sum_of_h = np.sum(self.tree[node_num].hess)
        weight = -sum_of_g / (sum_of_h + self.lambda_)

        self.tree[node_num].weight = weight

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


class GBDTTree(Tree):
    def __init__(self, max_depth, lambda_, gamma):
        super().__init__(max_depth, lambda_, gamma)

    def cal_gain(self, grad, hess, split_idx, indicator):
        left_grad = np.sum(grad[:split_idx])
        right_grad = np.sum(grad[split_idx:])
        left_indicator = np.sum(indicator[:split_idx])
        right_indicator = np.sum(indicator[split_idx:])

        return left_grad**2 / (
            left_indicator + self.lambda_) + right_grad**2 / (right_indicator +
                                                              self.lambda_)

    def set_weight(self, node_num):
        sum_of_g = np.sum(self.tree[node_num].grad)
        weight = -sum_of_g / (np.sum(self.tree[node_num].indicator) +
                              self.lambda_)

        self.tree[node_num].weight = weight

    def update_child(self, node_num, left_child, right_child):
        self.tree[2 * node_num +
                  1].grad = self.tree[node_num].grad * left_child
        self.tree[2 * node_num +
                  1].indicator = self.tree[node_num].indicator * left_child
        self.tree[2 * node_num +
                  2].grad = self.tree[node_num].grad * right_child
        self.tree[2 * node_num +
                  2].indicator = self.tree[node_num].indicator * right_child


class DecisionTree(Tree):
    def __init__(self, max_depth, lambda_, gamma):
        super().__init__(max_depth, lambda_, gamma)
        self.task_type = None  # ['classification', 'regression']

    def _gini(self, indicator, y):
        total_num = np.sum(indicator)
        positive_num = np.dot(indicator, y)
        negative_num = total_num - positive_num
        return 1 - (positive_num / total_num)**2 - (negative_num /
                                                    total_num)**2

    def _check_same_label(self, y, indicator):
        active_idx = np.nonzero(indicator)[0]
        active_y = y[active_idx]
        if np.sum(active_y) in [0, len(active_y)]:
            return True
        return False

    def cal_gini(self, split_idx, y, indicator):
        if self._check_same_label(y, indicator):
            # Return the maximum gini value
            return 1.0

        left_child_indicator = indicator * np.concatenate(
            [np.ones(split_idx),
             np.zeros(len(y) - split_idx)])
        right_child_indicator = indicator - left_child_indicator
        left_gini = self._gini(left_child_indicator, y)
        right_gini = self._gini(right_child_indicator, y)
        total_num = np.sum(indicator)
        return np.sum(left_child_indicator) / total_num * left_gini + sum(
            right_child_indicator) / total_num * right_gini

    def cal_sum_of_square_mean_err(self, split_idx, y, indicator):
        left_child_indicator = indicator * np.concatenate(
            [np.ones(split_idx),
             np.zeros(len(y) - split_idx)])
        right_child_indicator = indicator - left_child_indicator

        left_avg_value = np.dot(left_child_indicator,
                                y) / np.sum(left_child_indicator)
        right_avg_value = np.dot(right_child_indicator,
                                 y) / np.sum(right_child_indicator)
        return np.sum((y * indicator - left_avg_value * left_child_indicator -
                       right_avg_value * right_child_indicator)**2)

    def cal_gain(self, split_idx, y, indicator):
        if self.task_type == 'classification':
            return self.cal_gini(split_idx, y, indicator)
        elif self.task_type == 'regression':
            return self.cal_sum_of_square_mean_err(split_idx, y, indicator)
        else:
            raise ValueError(f'Task type error: {self.task_type}')

    def set_task_type(self, task_type):
        self.task_type = task_type

    def set_weight(self, node_num):
        active_idx = np.nonzero(self.tree[node_num].indicator)[0]
        active_y = self.tree[node_num].label[active_idx]

        # majority vote in classification
        if self.task_type == 'classification':
            vote = np.mean(active_y)
            self.tree[node_num].weight = 1 if vote >= 0.5 else 0
        # mean value for regression
        elif self.task_type == 'regression':
            self.tree[node_num].weight = np.mean(active_y)
        else:
            raise ValueError

    def update_child(self, node_num, left_child, right_child):
        self.tree[2 * node_num +
                  1].label = self.tree[node_num].label * left_child
        self.tree[2 * node_num +
                  1].indicator = self.tree[node_num].indicator * left_child
        self.tree[2 * node_num +
                  2].label = self.tree[node_num].label * right_child
        self.tree[2 * node_num +
                  2].indicator = self.tree[node_num].indicator * right_child


class MultipleXGBTrees(object):
    def __init__(self, max_depth, lambda_, gamma, num_of_trees):
        self.trees = [
            XGBTree(max_depth=max_depth, lambda_=lambda_, gamma=gamma)
            for _ in range(num_of_trees)
        ]
        self.num_of_trees = num_of_trees
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth

    def __getitem__(self, item):
        return self.trees[item]


class MultipleGBDTTrees(object):
    def __init__(self, max_depth, lambda_, gamma, num_of_trees):
        self.trees = [
            GBDTTree(max_depth=max_depth, lambda_=lambda_, gamma=gamma)
            for _ in range(num_of_trees)
        ]
        self.num_of_trees = num_of_trees
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth

    def __getitem__(self, item):
        return self.trees[item]


class RandomForest(object):
    def __init__(self, max_depth, lambda_, gamma, num_of_trees):
        self.trees = [
            DecisionTree(max_depth=max_depth, lambda_=lambda_, gamma=gamma)
            for _ in range(num_of_trees)
        ]
        self.num_of_trees = num_of_trees
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth

    def __getitem__(self, item):
        return self.trees[item]

    def set_task_type(self, criterion_type):
        if criterion_type == 'CrossEntropyLoss':
            task_type = 'classification'
        elif 'regression' in criterion_type.lower():
            task_type = 'regression'
        else:
            raise ValueError

        for tree in self.trees:
            tree.set_task_type(task_type)
