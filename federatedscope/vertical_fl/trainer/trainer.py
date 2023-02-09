import numpy as np
import logging
from collections import deque

from federatedscope.vertical_fl.dataloader.utils import batch_iter
from federatedscope.vertical_fl.loss.utils import get_vertical_loss

logger = logging.getLogger(__name__)


class VerticalTrainer(object):
    def __init__(self, model, data, device, config, monitor):
        self.model = model
        self.data = data
        self.device = device
        self.cfg = config
        self.monitor = monitor

        self.eta = config.train.optimizer.eta

        self.merged_feature_order = None
        self.client_feature_order = None
        self.extra_info = None
        self.batch_x = None
        self.batch_y = None
        self.batch_y_hat = None
        self.batch_z = None

    def prepare_for_train(self, index=None):
        self.dataloader = batch_iter(self.data['train'],
                                     self.cfg.dataloader.batch_size,
                                     shuffled=True)
        self.criterion = get_vertical_loss(
            self.cfg.criterion.type,
            cal_hess=(self.cfg.model.type == 'xgb_tree'))
        batch_index, self.batch_x, self.batch_y = self._fetch_train_data(index)
        feature_order_info = self._get_feature_order_info(self.batch_x)
        if 'raw_feature_order' in feature_order_info:
            # When applying protect method, the raw (real) feature order might
            # be different from the shared feature order
            self.client_feature_order = feature_order_info['raw_feature_order']
            feature_order_info.pop('raw_feature_order')
        else:
            self.client_feature_order = feature_order_info['feature_order']
        if index is None:
            self.batch_y_hat = np.random.uniform(low=0.0,
                                                 high=1.0,
                                                 size=len(self.batch_y))
            self.batch_z = 0
        return batch_index, feature_order_info

    def train(self, feature_order_info=None, tree_num=0, node_num=None):
        # Start to build a tree
        if node_num is None:
            if tree_num == 0 and feature_order_info is not None:
                self.merged_feature_order, self.extra_info = \
                    self._parse_feature_order(feature_order_info)
            return self._compute_for_root(tree_num=tree_num)
        # Continue training
        else:
            return self._compute_for_node(tree_num, node_num)

    def get_feature_value(self, feature_idx, value_idx):
        assert self.batch_x is not None

        instance_idx = self.client_feature_order[feature_idx][value_idx]
        return self.batch_x[instance_idx, feature_idx]

    def _predict(self, tree_num):
        self._compute_weight(tree_num, node_num=0)

    def _fetch_train_data(self, index=None):
        if index is None:
            return next(self.dataloader)
        else:
            return index, self.data['train']['x'][index], None

    def _parse_feature_order(self, feature_order_info):
        client_ids = list(feature_order_info.keys())
        client_ids = sorted(client_ids)
        merged_feature_order = np.concatenate(
            [feature_order_info[idx]['feature_order'] for idx in client_ids])

        # TODO: different extra_info for different clients
        extra_info = feature_order_info[client_ids[0]].get('extra_info', None)
        if extra_info is not None:
            merged_extra_info = dict()
            for each_key in extra_info.keys():
                merged_extra_info[each_key] = np.concatenate([
                    feature_order_info[idx]['extra_info'][each_key]
                    for idx in client_ids
                ])
        else:
            merged_extra_info = None

        return merged_feature_order, merged_extra_info

    def _get_feature_order_info(self, data):
        num_of_feature = data.shape[1]
        feature_order = [0] * num_of_feature
        for i in range(num_of_feature):
            feature_order[i] = data[:, i].argsort()
        return {'feature_order': feature_order}

    def _get_ordered_gh(self, tree_num, node_num, feature_idx):
        order = self.merged_feature_order[feature_idx]
        ordered_g = self.model[tree_num][node_num].grad[order]
        if self.model[tree_num][node_num].hess is None:
            # hess is not used in GBDT
            ordered_h = None
        else:
            ordered_h = self.model[tree_num][node_num].hess[order]
        return ordered_g, ordered_h

    def _get_best_gain(self, tree_num, node_num):
        best_gain = 0
        split_ref = {'feature_idx': None, 'value_idx': None}

        instance_num = self.batch_x.shape[0]
        feature_num = len(self.merged_feature_order)
        if self.extra_info is not None:
            split_position = self.extra_info.get(
                'split_position',
                [range(instance_num) for _ in range(feature_num)])
        else:
            # The left/right sub-tree cannot be empty
            split_position = [
                range(1, instance_num) for _ in range(feature_num)
            ]
        for feature_idx in range(feature_num):
            ordered_g, ordered_h = self._get_ordered_gh(
                tree_num, node_num, feature_idx)
            for value_idx in split_position[feature_idx]:
                gain = self.model[tree_num].cal_gain(ordered_g, ordered_h,
                                                     value_idx, node_num)

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
                split_feature = self.merged_feature_order[
                    split_ref['feature_idx']]
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
