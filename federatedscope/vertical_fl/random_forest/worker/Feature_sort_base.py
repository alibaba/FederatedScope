import time

import numpy as np
import pandas as pd

from federatedscope.core.message import Message


class Feature_sort_base:
    """
    This class contains the basic algorithm for vertical random forest, i.e.,
    the clients who do not hold labels will send their orders of all features
    to label owner
    """
    def __init__(self, obj):
        self.client = obj
        self.total_feature_order_dict = dict()
        self.total_ordered_label_list = [0] * self.client.total_num_of_feature

    def preparation(self):
        self.client.register_handlers('feature_order',
                                      self.callback_func_for_feature_order)
        self.client.register_handlers('split', self.callback_func_for_split)

        self.client.order_feature(self.client.x)
        if not self.client.own_label:
            self.client.comm_manager.send(
                Message(msg_type='feature_order',
                        sender=self.client.ID,
                        state=self.client.state,
                        receiver=self.client.num_of_parties,
                        content=self.client.feature_order))

    # label owner
    def callback_func_for_feature_order(self, message: Message):
        feature_order = message.content
        self.total_feature_order_dict[message.sender - 1] = feature_order
        if len(self.total_feature_order_dict
               ) == self.client.num_of_parties - 1:
            tree_num = self.client.state
            self.total_feature_order_dict[self.client.ID -
                                          1] = self.client.feature_order
            sorted_list = sorted(self.total_feature_order_dict.items(),
                                 key=lambda x: x[0])
            self.total_feature_order_dict = dict(sorted_list)

            self.total_feature_order_list = np.concatenate(
                list(self.total_feature_order_dict.values()))
            self.total_feature_order_dict = dict()
            self.compute_for_root(tree_num)

    # label owner
    def compute_for_root(self, tree_num):
        node_num = 0
        self.client.tree_list[tree_num][node_num].label = self.client.y
        self.client.tree_list[tree_num][node_num].indicator = np.ones(
            len(self.client.y))
        self.order_act_on_label(tree_num, node_num)
        self.compute_for_node(tree_num, node_num)

    # label owner
    def compute_for_node(self, tree_num, node_num):
        # if node_num < 2 ** self.client.max_tree_depth - 1:
        #    print(self.client.tree_list[tree_num][node_num].indicator)
        if node_num >= 2**self.client.max_tree_depth - 1:
            self.client.prediction(tree_num)
        elif self.client.tree_list[tree_num][node_num].status == 'off':
            self.compute_for_node(tree_num, node_num + 1)
        elif node_num >= 2**(self.client.max_tree_depth - 1) - 1:
            self.client.set_weight(tree_num, node_num)
        else:
            # self.order_act_on_label(tree_num, node_num)
            split_ref = {'feature_idx': None, 'value_idx': None}
            flag = 0
            nonzero_idx = np.nonzero(
                self.client.tree_list[tree_num][node_num].indicator)[0]
            if self.client.criterion_type == 'CrossEntropyLoss':
                best_gini = 1
                for feature_idx in range(self.client.total_num_of_feature):
                    # for value_idx in range(self.client.x.shape[0]):
                    if len(nonzero_idx) == 0:
                        continue
                    for value_idx in range(nonzero_idx[0],
                                           nonzero_idx[-1] + 1):
                        left = np.concatenate(
                            (np.ones(value_idx),
                             np.zeros(self.client.x.shape[0] - value_idx)))
                        gini = self.client.cal_gini(
                            left, self.client.tree_list[tree_num]
                            [node_num].indicator,
                            self.total_ordered_label_list[feature_idx])
                        if gini < best_gini:
                            best_gini = gini
                            split_ref['feature_idx'] = feature_idx
                            split_ref['value_idx'] = value_idx

                if best_gini < 1:
                    print(tree_num, node_num, best_gini, split_ref)
                    flag = 1
            elif self.client.criterion_type == 'Regression':
                best_measure = float('inf')
                for feature_idx in range(self.client.total_num_of_feature):
                    if len(nonzero_idx) == 0:
                        continue
                    for value_idx in range(nonzero_idx[0],
                                           nonzero_idx[-1] + 1):
                        # for value_idx in range(self.client.x.shape[0]):
                        left = np.concatenate(
                            (np.ones(value_idx),
                             np.zeros(self.client.x.shape[0] - value_idx)))
                        measure = self.client.sse(
                            left, self.client.tree_list[tree_num]
                            [node_num].indicator,
                            self.total_ordered_label_list[feature_idx])
                        if measure < best_measure:
                            best_measure = measure
                            split_ref['feature_idx'] = feature_idx
                            split_ref['value_idx'] = value_idx
                if best_measure < float('inf'):
                    print(tree_num, node_num, best_measure, split_ref)
                    flag = 1
            if flag == 1:
                split_feature = self.total_feature_order_list[
                    split_ref['feature_idx']]
                left_child_idx = np.zeros(len(self.client.y))
                for x in range(split_ref['value_idx']):
                    left_child_idx[split_feature[x]] = 1
                right_child_idx = np.ones(len(self.client.y)) - left_child_idx

                self.client.tree_list[tree_num][
                    2 * node_num + 1].label = self.client.tree_list[tree_num][
                        node_num].label * left_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 1].indicator = self.client.tree_list[
                        tree_num][node_num].indicator * left_child_idx
                self.client.tree_list[tree_num][
                    2 * node_num + 2].label = self.client.tree_list[tree_num][
                        node_num].label * right_child_idx
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

    def callback_func_for_split(self, message: Message):
        tree_num, node_num, split_ref = message.content
        feature_idx = split_ref['feature_idx'] - self.client.feature_list[
            self.client.ID - 1]
        self.client.feature_importance[feature_idx] += 1
        value_idx = split_ref['value_idx']
        # feature_value = sorted(self.client.x[:, feature_idx])[value_idx]
        feature_value = self.client.x[:, feature_idx][
            self.client.feature_order[feature_idx][value_idx]]
        # print(feature_idx, feature_value)
        self.client.tree_list[tree_num][node_num].feature_idx = feature_idx
        self.client.tree_list[tree_num][node_num].feature_value = feature_value

        self.client.comm_manager.send(
            Message(msg_type='compute_next_node',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.num_of_parties,
                    content=(tree_num, node_num)))

    def perm_act_on_list(self, pi, list):
        res = np.zeros(len(list))
        for i in range(len(list)):
            res[i] = list[pi[i]]
        return res

    def order_act_on_label(self, tree_num, node_num):
        self.total_ordered_lable_list = [0] * self.client.total_num_of_feature
        for i in range(self.client.total_num_of_feature):
            self.total_ordered_label_list[i] = self.perm_act_on_list(
                self.total_feature_order_list[i],
                self.client.tree_list[tree_num][node_num].label)
