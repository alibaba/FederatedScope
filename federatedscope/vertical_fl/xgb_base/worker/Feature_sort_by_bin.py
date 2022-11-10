import time

import numpy as np
import pandas as pd
import collections

from federatedscope.core.message import Message


class Feature_sort_by_bin:
    """
        This class contains the bin algorithm for xgboost, i.e.,
        the clients who do not hold labels will first get their orders
        of all features, and then partition each order to several bins,
        in each bin, they can do some permutation to protect their privacy.
        """
    def __init__(self, obj, bin_num=100):
        self.client = obj
        self.total_feature_order_dict = dict()
        self.bin_num = bin_num
        self.total_feature_order_list_of_dict = dict()
        self.feature_order_list_of_dict = [
            dict() for _ in range(self.client.my_num_of_feature)
        ]

    def partition_to_bin(self, ordered_list):
        bin_size = int(np.ceil(self.client.batch_size / self.bin_num))
        for i in range(len(ordered_list)):
            for j in range(self.bin_num):
                self.feature_order_list_of_dict[i][j] = ordered_list[i][
                    j * bin_size:(j + 1) * bin_size]
                # TODO: add some perturbation in each set

    def preparation(self):
        self.client.register_handlers('feature_order',
                                      self.callback_func_for_feature_order)
        self.client.register_handlers('split', self.callback_func_for_split)

        self.client.order_feature(self.client.x)
        self.partition_to_bin(self.client.feature_order)

        if not self.client.own_label:
            self.client.comm_manager.send(
                Message(msg_type='feature_order',
                        sender=self.client.ID,
                        state=self.client.state,
                        receiver=self.client.num_of_parties,
                        content=self.feature_order_list_of_dict))

    # label owner
    def callback_func_for_feature_order(self, message: Message):
        feature_order_list_of_dict = message.content
        self.total_feature_order_dict[message.sender -
                                      1] = feature_order_list_of_dict
        if len(self.total_feature_order_dict
               ) == self.client.num_of_parties - 1:
            tree_num = 0
            self.total_feature_order_dict[self.client.ID -
                                          1] = self.feature_order_list_of_dict

            sorted_list = sorted(self.total_feature_order_dict.items(),
                                 key=lambda x: x[0])
            self.total_feature_order_dict = dict(sorted_list)

            self.total_feature_order_list_of_dict = np.concatenate(
                list(self.total_feature_order_dict.values()))
            self.total_feature_order_dict = dict()
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

    def bin_act_on_list(self, bin, list):
        length = len(bin)
        res = np.zeros(length)
        for i in range(length):
            tmp = 0
            for j in bin[i]:
                tmp += list[j]
            res[i] = tmp
        return res

    def bin_act_on_gh(self, tree_num, node_num):
        self.client.total_ordered_g_bin_list = [
            0
        ] * self.client.total_num_of_feature
        self.client.total_ordered_h_bin_list = [
            0
        ] * self.client.total_num_of_feature

        for i in range(self.client.total_num_of_feature):
            self.client.total_ordered_g_bin_list[i] = self.bin_act_on_list(
                self.total_feature_order_list_of_dict[i],
                self.client.tree_list[tree_num][node_num].grad)
            self.client.total_ordered_h_bin_list[i] = self.bin_act_on_list(
                self.total_feature_order_list_of_dict[i],
                self.client.tree_list[tree_num][node_num].hess)

    # label owner
    def compute_for_node(self, tree_num, node_num):
        if node_num >= 2**self.client.max_tree_depth - 1:
            self.client.prediction(tree_num)
        elif self.client.tree_list[tree_num][node_num].status == 'off':
            self.compute_for_node(tree_num, node_num + 1)
        elif node_num >= 2**(self.client.max_tree_depth - 1) - 1:
            self.client.set_weight(tree_num, node_num)
        else:
            self.bin_act_on_gh(tree_num, node_num)
            best_gain = 0
            split_ref = {'feature_idx': None, 'bin_idx': None}

            for feature_idx in range(self.client.total_num_of_feature):
                for bin_idx in range(self.bin_num):
                    left_grad = np.sum(
                        self.client.total_ordered_g_bin_list[feature_idx]
                        [:bin_idx])
                    right_grad = np.sum(
                        self.client.total_ordered_g_bin_list[feature_idx]
                    ) - left_grad
                    left_hess = np.sum(
                        self.client.total_ordered_h_bin_list[feature_idx]
                        [:bin_idx])
                    right_hess = np.sum(
                        self.client.total_ordered_h_bin_list[feature_idx]
                    ) - left_hess
                    gain = self.client.cal_gain(left_grad, right_grad,
                                                left_hess, right_hess)

                    if gain > best_gain:
                        best_gain = gain
                        split_ref['feature_idx'] = feature_idx
                        split_ref['bin_idx'] = bin_idx

            if best_gain > 0:
                split_feature = self.total_feature_order_list_of_dict[
                    split_ref['feature_idx']]
                bin_idx = split_ref['bin_idx']
                left_child_idx = np.zeros(len(self.client.y))

                for x in range(bin_idx):
                    for i in split_feature[x]:
                        left_child_idx[i] = 1

                right_child_idx = np.ones(len(self.client.y)) - left_child_idx

                self.client.tree_list[tree_num][
                    2 * node_num + 1].grad = self.client.tree_list[tree_num][
                        node_num].grad * left_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 1].hess = self.client.tree_list[tree_num][
                        node_num].hess * left_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 1].indicator = self.client.tree_list[
                        tree_num][node_num].indicator * left_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 2].grad = self.client.tree_list[tree_num][
                        node_num].grad * right_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 2].hess = self.client.tree_list[tree_num][
                        node_num].hess * right_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 2].indicator = self.client.tree_list[
                        tree_num][node_num].indicator * right_child_idx

                for i in range(self.client.num_of_parties):
                    if self.client.feature_list[i] <= split_ref[
                            'feature_idx'] < self.client.feature_list[i + 1]:
                        self.client.tree_list[tree_num][
                            node_num].member = i + 1
                        self.client.tree_list[tree_num][
                            node_num].feature_idx = split_ref[
                                'feature_idx'] - self.client.feature_list[i]
                        self.client.comm_manager.send(
                            Message(msg_type='split',
                                    sender=self.client.ID,
                                    state=self.client.state,
                                    receiver=i + 1,
                                    content=(tree_num, node_num, split_ref)))
                        break
            else:
                self.client.set_weight(tree_num, node_num)

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

    def callback_func_for_split(self, message: Message):
        tree_num, node_num, split_ref = message.content
        feature_idx = split_ref['feature_idx'] - self.client.feature_list[
            self.client.ID - 1]
        bin_idx = split_ref['bin_idx']
        # feature_value = sorted(self.x[:, feature_idx])[value_idx]
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
            Message(msg_type='compute_next_node',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.num_of_parties,
                    content=(tree_num, node_num)))
