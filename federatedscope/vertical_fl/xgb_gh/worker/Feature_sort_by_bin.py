import time

import numpy as np
import pandas as pd
import collections

from federatedscope.core.message import Message


class Feature_sort_by_bin:
    """
        This class contains the bin algorithm for xgboost, i.e.,
        the clients who do not hold labels will first get their orders
        of all features, and then partition each order to several bins.
        When receive g and h from label-owner, they can compute local
        best gain efficiently.
        """
    def __init__(self, obj, bin_num=100):
        self.client = obj
        self.total_feature_order_dict = dict()
        self.bin_num = bin_num
        self.total_feature_order_list_of_dict = dict()
        self.feature_order_list_of_dict = [
            dict() for _ in range(self.client.my_num_of_feature)
        ]
        self.gain_dict = dict()

    def order_partition_to_bin(self, ordered_list):
        bin_size = int(np.ceil(self.client.batch_size / self.bin_num))
        for i in range(len(ordered_list)):
            for j in range(self.bin_num):
                self.feature_order_list_of_dict[i][j] = ordered_list[i][
                    j * bin_size:(j + 1) * bin_size]

    def preparation(self):
        self.client.register_handlers('gh', self.callback_func_for_gh)
        self.client.register_handlers('local_best_gain',
                                      self.callback_func_for_local_best_gain)
        self.client.register_handlers('split', self.callback_func_for_split)
        self.client.register_handlers('split_LR',
                                      self.callback_func_for_split_LR)

        self.client.order_feature(self.client.x)
        self.order_partition_to_bin(self.client.feature_order)

        if self.client.own_label:
            tree_num = 0
            self.compute_for_root(tree_num)

    # label owner
    def compute_for_root(self, tree_num):
        g, h = self.client.ls.get_grad_and_hess(self.client.y,
                                                self.client.y_hat)
        node_num = 0
        self.client.tree_list[tree_num][node_num].grad = g
        self.client.tree_list[tree_num][node_num].hess = h
        self.client.tree_list[tree_num][node_num].indicator = np.ones(
            len(self.client.y))
        self.compute_for_node(tree_num, node_num)

    # label owner
    def compute_for_node(self, tree_num, node_num):
        if node_num >= 2**self.client.max_tree_depth - 1:
            self.client.prediction(tree_num)
        elif self.client.tree_list[tree_num][node_num].status == 'off':
            self.compute_for_node(tree_num, node_num + 1)
        elif node_num >= 2**(self.client.max_tree_depth - 1) - 1:
            self.client.set_weight(tree_num, node_num)
        else:
            g = self.client.tree_list[tree_num][node_num].grad
            h = self.client.tree_list[tree_num][node_num].hess
            self.client.comm_manager.send(
                Message(msg_type='gh',
                        sender=self.client.ID,
                        state=self.client.state,
                        receiver=[
                            each for each in list(
                                self.client.comm_manager.neighbors.keys())
                            if each != self.client.server_id
                        ],
                        content=(tree_num, node_num, g, h)))
            self.compute_local_best_gain(tree_num, node_num, g, h)

    def callback_func_for_gh(self, message: Message):
        tree_num, node_num, g, h = message.content
        self.compute_local_best_gain(tree_num, node_num, g, h)

    def compute_local_best_gain(self, tree_num, node_num, g, h):
        self.order_act_on_gh(g, h)
        self.ordered_part_sum_g = self.partition_to_bin(self.ordered_list_g)
        self.ordered_part_sum_h = self.partition_to_bin(self.ordered_list_h)
        best_gain = 0
        self.split_ref = {'feature_idx': None, 'bin_idx': None}

        for feature_idx in range(self.client.my_num_of_feature):
            for bin_idx in range(self.bin_num):
                left_grad = np.sum(
                    self.ordered_part_sum_g[feature_idx][:bin_idx])
                right_grad = np.sum(
                    self.ordered_part_sum_g[feature_idx]) - left_grad
                left_hess = np.sum(
                    self.ordered_part_sum_h[feature_idx][:bin_idx])
                right_hess = np.sum(
                    self.ordered_part_sum_h[feature_idx]) - left_hess
                gain = self.client.cal_gain(left_grad, right_grad, left_hess,
                                            right_hess)

                if gain > best_gain:
                    best_gain = gain
                    self.split_ref['feature_idx'] = feature_idx
                    self.split_ref['bin_idx'] = bin_idx
        self.client.comm_manager.send(
            Message(msg_type='local_best_gain',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.num_of_parties,
                    content=(tree_num, node_num, best_gain)))

    # label owner
    def callback_func_for_local_best_gain(self, message: Message):
        tree_num, node_num, local_best_gain = message.content
        self.gain_dict[message.sender - 1] = local_best_gain
        if len(self.gain_dict) == self.client.num_of_parties:
            client_idx = max(self.gain_dict, key=self.gain_dict.get)
            if self.gain_dict[client_idx] == 0:
                self.client.set_weight(tree_num, node_num)
            else:
                self.client.tree_list[tree_num][
                    node_num].member = client_idx + 1
                self.client.comm_manager.send(
                    Message(msg_type='split',
                            sender=self.client.ID,
                            state=self.client.state,
                            receiver=client_idx + 1,
                            content=(tree_num, node_num)))
            self.gain_dict = dict()

    def callback_func_for_split(self, message: Message):
        tree_num, node_num = message.content

        feature_idx = self.split_ref['feature_idx']
        bin_idx = self.split_ref['bin_idx']

        left_child_idx = np.zeros(len(self.client.x))
        for i in range(bin_idx):
            for x in self.feature_order_list_of_dict[feature_idx][i]:
                left_child_idx[x] = 1

        right_child_idx = np.ones(len(self.client.x)) - left_child_idx

        self.client.feature_importance[feature_idx] += 1
        if bin_idx == 0:
            feature_value = self._min(
                self.client.x[:, feature_idx],
                self.feature_order_list_of_dict[feature_idx][bin_idx])
        elif bin_idx == self.bin_num - 1:
            feature_value = self._max(
                self.client.x[:, feature_idx],
                self.feature_order_list_of_dict[feature_idx][bin_idx])
        else:
            min_num = self._min(
                self.client.x[:, feature_idx],
                self.feature_order_list_of_dict[feature_idx][bin_idx + 1])
            max_num = self._max(
                self.client.x[:, feature_idx],
                self.feature_order_list_of_dict[feature_idx][bin_idx])
            feature_value = (max_num + min_num) / 2

        self.client.tree_list[tree_num][node_num].feature_idx = feature_idx
        self.client.tree_list[tree_num][node_num].feature_value = feature_value
        self.client.comm_manager.send(
            Message(msg_type='split_LR',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.num_of_parties,
                    content=(tree_num, node_num, left_child_idx,
                             right_child_idx)))

    def callback_func_for_split_LR(self, message: Message):
        tree_num, node_num, left_child_idx, right_child_idx = message.content
        self.client.tree_list[tree_num][
            2 * node_num + 1].grad = self.client.tree_list[tree_num][
                node_num].grad * left_child_idx
        self.client.tree_list[tree_num][
            2 * node_num + 1].hess = self.client.tree_list[tree_num][
                node_num].hess * left_child_idx
        self.client.tree_list[tree_num][
            2 * node_num + 1].indicator = self.client.tree_list[tree_num][
                node_num].indicator * left_child_idx
        self.client.tree_list[tree_num][
            2 * node_num + 2].grad = self.client.tree_list[tree_num][
                node_num].grad * right_child_idx
        self.client.tree_list[tree_num][
            2 * node_num + 2].hess = self.client.tree_list[tree_num][
                node_num].hess * right_child_idx
        self.client.tree_list[tree_num][
            2 * node_num + 2].indicator = self.client.tree_list[tree_num][
                node_num].indicator * right_child_idx

        self.compute_for_node(tree_num, node_num + 1)

    def partition_to_bin(self, ordered_list):
        bin_size = int(np.ceil(self.client.batch_size / self.bin_num))
        res = [[0] * self.bin_num for _ in range(len(ordered_list))]
        for i in range(len(ordered_list)):
            for j in range(self.bin_num):
                res[i][j] = sum(ordered_list[i][j * bin_size:(j + 1) *
                                                bin_size])
        return res

    def order_act_on_gh(self, g, h):
        self.ordered_list_g = [0] * self.client.my_num_of_feature
        self.ordered_list_h = [0] * self.client.my_num_of_feature
        for i in range(self.client.my_num_of_feature):
            self.ordered_list_g[i] = self.perm_act_on_list(
                self.client.feature_order[i], g)
            self.ordered_list_h[i] = self.perm_act_on_list(
                self.client.feature_order[i], h)

    def perm_act_on_list(self, pi, list):
        res = np.zeros(len(list))
        for i in range(len(list)):
            res[i] = list[pi[i]]
        return res

    def _min(self, val_list, ind_list):
        res = float('inf')
        for i in ind_list:
            res = min(res, val_list[i])
        return res

    def _max(self, val_list, ind_list):
        res = -float('inf')
        for i in ind_list:
            res = max(res, val_list[i])
        return res
