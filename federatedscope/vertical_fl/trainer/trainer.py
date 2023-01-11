import numpy as np
import logging
from collections import deque

from federatedscope.vertical_fl.dataloader.utils import batch_iter
from federatedscope.vertical_fl.loss.utils import get_vertical_loss

logger = logging.getLogger(__name__)


class VerticalTrainer(object):
    def __init__(self, model, data, device, config, monitor, only_for_eval):
        self.model = model
        self.data = data
        self.device = device
        self.cfg = config
        self.monitor = monitor
        self.only_for_eval = only_for_eval

        self.bin_num = config.train.optimizer.bin_num
        self.eta = config.train.optimizer.eta

        self.dataloader = batch_iter(self.data['train'],
                                     self.cfg.dataloader.batch_size,
                                     shuffled=True)
        self.batch_x = None
        self.batch_y = None
        self.batch_y_hat = None
        self.batch_z = None

    def prepare_for_train(self, index=None):
        self.criterion = get_vertical_loss(self.cfg.criterion.type)
        batch_index, self.batch_x, self.batch_y = self._fetch_train_data(index)
        feature_order = self._get_feature_order(self.batch_x)
        if index is None:
            self.batch_y_hat = np.random.uniform(low=0.0,
                                                 high=1.0,
                                                 size=len(self.batch_y))
            self.batch_z = 0
        return batch_index, feature_order

    def train(self, feature_order=None, tree_num=0, node_num=None):
        # Start to build a tree
        if node_num is None:
            if tree_num == 0 and feature_order is not None:
                self.feature_order = feature_order
            return self._compute_for_root(tree_num=tree_num)
        # Continue training
        else:
            return self._compute_for_node(tree_num, node_num)

    def _predict(self, tree_num):
        self._compute_weight(tree_num, node_num=0)

    def _fetch_train_data(self, index=None):
        if index is None:
            return next(self.dataloader)
        else:
            return index, self.data['train']['x'][index], None

    def _get_feature_order(self, data):
        num_of_feature = data.shape[1]
        feature_order = [0] * num_of_feature
        for i in range(num_of_feature):
            feature_order[i] = data[:, i].argsort()
        return feature_order

    def _get_ordered_gh(self, tree_num, node_num, feature_idx):
        order = self.feature_order[feature_idx]
        ordered_g = self.model[tree_num][node_num].grad[order]
        ordered_h = self.model[tree_num][node_num].hess[order]
        return ordered_g, ordered_h

    def _get_best_gain(self, tree_num, node_num):
        best_gain = 0
        split_ref = {'feature_idx': None, 'value_idx': None}

        instance_num = self.batch_x.shape[0]
        feature_num = len(self.feature_order)
        for feature_idx in range(feature_num):
            ordered_g, ordered_h = self._get_ordered_gh(
                tree_num, node_num, feature_idx)
            for value_idx in range(instance_num):
                gain = self.model[tree_num].cal_gain(ordered_g, ordered_h,
                                                     value_idx)

                if gain > best_gain:
                    best_gain = gain
                    split_ref['feature_idx'] = feature_idx
                    split_ref['value_idx'] = value_idx

        return best_gain, split_ref

    def _compute_for_root(self, tree_num):
        g, h = self.criterion.get_grad_and_hess(self.batch_y, self.batch_y_hat)
        node_num = 0
        self.model[tree_num][node_num].grad = g
        self.model[tree_num][node_num].hess = h
        self.model[tree_num][node_num].indicator = np.ones(len(self.batch_y))
        return self._compute_for_node(tree_num, node_num=node_num)

    def _compute_for_node(self, tree_num, node_num):

        # All the nodes have been traversed
        if node_num >= 2**self.model.max_depth - 1:
            self._predict(tree_num)
            finish_flag = True
            return finish_flag, None
        elif self.model[tree_num][node_num].status == 'off':
            return self._compute_for_node(tree_num, node_num + 1)
        # The leaf node
        elif node_num >= 2**(self.model.max_depth - 1) - 1:
            self._set_weight_and_status(tree_num, node_num)
            return self._compute_for_node(tree_num, node_num + 1)
        # Calculate best gain
        else:
            best_gain, split_ref = self._get_best_gain(tree_num, node_num)
            if best_gain > 0:
                split_feature = self.feature_order[split_ref['feature_idx']]
                left_child = np.zeros(self.batch_x.shape[0])
                for x in range(split_ref['value_idx']):
                    left_child[split_feature[x]] = 1
                right_child = np.ones(self.batch_x.shape[0]) - left_child
                self.model[tree_num].update_child(node_num, left_child,
                                                  right_child)

                finish_flag = False
                results = (split_ref, tree_num, node_num)
                return finish_flag, results
            else:
                self._set_weight_and_status(tree_num, node_num)
                return self._compute_for_node(tree_num, node_num + 1)

    def _compute_weight(self, tree_num, node_num):
        if node_num >= 2**self.model.max_depth - 1:
            if tree_num == 0:
                self.batch_y_hat = self.batch_z
            else:
                self.batch_y_hat += self.batch_z
            self.batch_z = 0

        else:
            if self.model[tree_num][node_num].weight:
                self.batch_z += self.model[tree_num][
                    node_num].weight * self.model[tree_num][
                        node_num].indicator * self.eta
            self._compute_weight(tree_num, node_num + 1)

    def _set_weight_and_status(self, tree_num, node_num):
        self.model[tree_num].set_weight(node_num)

        queue = deque()
        queue.append(node_num)
        while len(queue) > 0:
            cur_node = queue.popleft()
            self.model[tree_num].set_status(cur_node, status='off')
            if 2 * cur_node + 2 <= 2**self.model[tree_num].max_depth - 1:
                queue.append(2 * cur_node + 1)
                queue.append(2 * cur_node + 2)
