import numpy as np
import logging
from collections import deque

from federatedscope.vertical_fl.dataloader.utils import VerticalDataSampler
from federatedscope.vertical_fl.loss.utils import get_vertical_loss

logger = logging.getLogger(__name__)


class VerticalTrainer(object):
    def __init__(self, model, data, device, config, monitor):
        self.model = model
        self.data = data
        self.device = device
        self.cfg = config
        self.monitor = monitor

        self.merged_feature_order = None
        self.client_feature_order = None
        self.complete_feature_order_info = None
        self.client_feature_num = list()
        self.extra_info = None
        self.client_extra_info = None
        self.batch_x = None
        self.batch_y = None
        self.batch_y_hat = None
        self.batch_z = 0

    def _init_for_train(self):
        self.eta = self.cfg.train.optimizer.eta
        self.dataloader = VerticalDataSampler(
            data=self.data['train'],
            use_full_trainset=True,
            feature_frac=self.cfg.vertical.feature_subsample_ratio)
        self.criterion = get_vertical_loss(loss_type=self.cfg.criterion.type,
                                           model_type=self.cfg.model.type)

    def prepare_for_train(self):
        if self.dataloader.use_full_trainset:
            complete_feature_order_info = self._get_feature_order_info(
                self.data['train']['x'])
            self.complete_feature_order_info = complete_feature_order_info
        else:
            self.complete_feature_order_info = None

    def fetch_train_data(self, index=None):
        # Clear the variables for last training round
        self.client_feature_num.clear()

        # Fetch new data
        batch_index, self.batch_x, self.batch_y = self.dataloader.sample_data(
            sample_size=self.cfg.dataloader.batch_size, index=index)
        feature_index, self.batch_x = self.dataloader.sample_feature(
            self.batch_x)

        # If the complete trainset is used, we only need to get the slices
        # from the complete feature order info according to the feature index,
        # rather than re-ordering the instance
        if self.dataloader.use_full_trainset:
            assert self.complete_feature_order_info is not None
            feature_order_info = dict()
            for key in self.complete_feature_order_info:
                if isinstance(self.complete_feature_order_info[key],
                              list) or isinstance(
                                  self.complete_feature_order_info[key],
                                  np.ndarray):
                    feature_order_info[key] = [
                        self.complete_feature_order_info[key][_index]
                        for _index in feature_index
                    ]
                else:
                    feature_order_info[key] = self.complete_feature_order_info[
                        key]
        else:
            feature_order_info = self._get_feature_order_info(self.batch_x)

        if 'raw_feature_order' in feature_order_info:
            # When applying protect method, the raw (real) feature order might
            # be different from the shared feature order
            self.client_feature_order = feature_order_info['raw_feature_order']
            feature_order_info.pop('raw_feature_order')
        else:
            self.client_feature_order = feature_order_info['feature_order']
            self.client_extra_info = feature_order_info.get('extra_info', None)

        return batch_index, feature_order_info

    def train(self, training_info=None, tree_num=0, node_num=None):
        # Start to build a tree
        if node_num is None:
            if training_info is not None and \
                    self.cfg.vertical.mode == 'order_based':
                self.merged_feature_order, self.extra_info = \
                    self._parse_training_info(training_info)
            return self._compute_for_root(tree_num=tree_num)
        # Continue training
        else:
            return self._compute_for_node(tree_num, node_num)

    def get_abs_feature_idx(self, rel_feature_idx):
        if self.dataloader.selected_feature_index is None:
            return rel_feature_idx
        else:
            return self.dataloader.selected_feature_index[rel_feature_idx]

    def get_feature_value(self, feature_idx, value_idx):
        assert self.batch_x is not None

        instance_idx = self.client_feature_order[feature_idx][value_idx]
        return self.batch_x[instance_idx, feature_idx]

    def _predict(self, tree_num):
        self._compute_weight(tree_num, node_num=0)

    def _parse_training_info(self, feature_order_info):
        client_ids = list(feature_order_info.keys())
        client_ids = sorted(client_ids)
        merged_feature_order = list()
        for each_client in client_ids:
            _feature_order = feature_order_info[each_client]['feature_order']
            merged_feature_order.append(_feature_order)
            self.client_feature_num.append(len(_feature_order))
        merged_feature_order = np.concatenate(merged_feature_order)

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

    def _get_ordered_gh(self,
                        tree_num,
                        node_num,
                        feature_idx,
                        grad=None,
                        hess=None,
                        indicator=None):
        order = self.merged_feature_order[feature_idx]
        if grad is not None:
            ordered_g = np.asarray(grad)[order]
        else:
            ordered_g = self.model[tree_num][node_num].grad[order]

        if hess is not None:
            ordered_h = np.asarray(hess)[order]
        elif self.model[tree_num][node_num].hess is not None:
            ordered_h = self.model[tree_num][node_num].hess[order]
        else:
            ordered_h = None

        if indicator is not None:
            ordered_indicator = np.asarray(indicator)[order]
        elif self.model[tree_num][node_num].indicator is not None:
            ordered_indicator = self.model[tree_num][node_num].indicator[order]
        else:
            ordered_indicator = None

        return ordered_g, ordered_h, ordered_indicator

    def _get_best_gain(self,
                       tree_num,
                       node_num,
                       grad=None,
                       hess=None,
                       indicator=None):
        best_gain = 0
        split_ref = {'feature_idx': None, 'value_idx': None}

        if self.merged_feature_order is None:
            self.merged_feature_order = self.client_feature_order
        if self.extra_info is None:
            self.extra_info = self.client_extra_info

        feature_num = len(self.merged_feature_order)
        split_position = None
        if self.extra_info is not None:
            split_position = self.extra_info.get('split_position', None)

        if self.model[tree_num][node_num].indicator is not None:
            activate_idx = [
                np.nonzero(self.model[tree_num][node_num].indicator[order])[0]
                for order in self.merged_feature_order
            ]
        else:
            activate_idx = [
                np.arange(self.batch_x.shape[0])
                for _ in self.merged_feature_order
            ]

        activate_idx = np.asarray(activate_idx)
        if split_position is None:
            # The left/right sub-tree cannot be empty
            split_position = activate_idx[:, 1:]

        for feature_idx in range(feature_num):
            ordered_g, ordered_h, ordered_indicator = self._get_ordered_gh(
                tree_num, node_num, feature_idx, grad, hess, indicator)
            order = self.merged_feature_order[feature_idx]
            for value_idx in split_position[feature_idx]:
                if self.model[tree_num].check_empty_child(
                        node_num, value_idx, order):
                    continue
                gain = self.model[tree_num].cal_gain(ordered_g, ordered_h,
                                                     value_idx,
                                                     ordered_indicator)

                if gain > best_gain:
                    best_gain = gain
                    split_ref['feature_idx'] = feature_idx
                    split_ref['value_idx'] = value_idx

        return best_gain > 0, split_ref, best_gain

    def _compute_for_root(self, tree_num):
        if self.batch_y_hat is None:
            # Assign a random predictions when tree_num = 0
            self.batch_y_hat = [
                np.random.uniform(low=0.0, high=1.0, size=len(self.batch_y))
            ]
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
            return 'train_finish', None
        elif self.model[tree_num][node_num].status == 'off':
            return self._compute_for_node(tree_num, node_num + 1)
        # The leaf node
        elif node_num >= 2**(self.model.max_depth - 1) - 1:
            self._set_weight_and_status(tree_num, node_num)
            return self._compute_for_node(tree_num, node_num + 1)
        # Calculate best gain
        else:
            if self.cfg.vertical.mode == 'order_based':
                improved_flag, split_ref, _ = self._get_best_gain(
                    tree_num, node_num)
                if improved_flag:
                    split_feature = self.merged_feature_order[
                        split_ref['feature_idx']]
                    left_child, right_child = self.get_children_indicator(
                        value_idx=split_ref['value_idx'],
                        split_feature=split_feature)
                    self.update_child(tree_num, node_num, left_child,
                                      right_child)
                    results = (split_ref, tree_num, node_num)
                    return 'call_for_node_split', results
                else:
                    self._set_weight_and_status(tree_num, node_num)
                    return self._compute_for_node(tree_num, node_num + 1)
            elif self.cfg.vertical.mode == 'label_based':
                results = (self.model[tree_num][node_num].grad,
                           self.model[tree_num][node_num].hess, tree_num,
                           node_num)
                return 'call_for_local_gain', results

    def _compute_weight(self, tree_num, node_num):
        if node_num >= 2**self.model.max_depth - 1:
            if tree_num == 0:
                self.batch_y_hat = [self.batch_z]
            else:
                self.batch_y_hat.append(self.batch_z)
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

    def get_children_indicator(self, value_idx, split_feature):
        left_child = np.zeros(self.batch_x.shape[0])
        for x in range(value_idx):
            left_child[split_feature[x]] = 1
        right_child = np.ones(self.batch_x.shape[0]) - left_child

        return left_child, right_child

    def update_child(self, tree_num, node_num, left_child, right_child):
        self.model[tree_num].update_child(node_num, left_child, right_child)

    def get_best_gain_from_msg(self, msg, tree_num=None, node_num=None):
        client_has_max_gain = None
        max_gain = None
        for client_id, local_gain in msg.items():
            gain, improved_flag, _ = local_gain
            if improved_flag:
                if max_gain is None or gain > max_gain:
                    max_gain = gain
                    client_has_max_gain = client_id

        return max_gain, client_has_max_gain, None
