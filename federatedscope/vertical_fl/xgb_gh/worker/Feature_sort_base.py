import time

import numpy as np
import pandas as pd

from federatedscope.core.message import Message


class Feature_sort_base:
    """
    This class contains the basic algorithm for xgboost, i.e.,
    the label-owner will send g and h to other clients
    """
    def __init__(self, obj):
        self.client = obj
        self.split_ref = dict()
        if self.client.own_label:
            self.gain_dict = dict()

    def preparation(self):
        self.client.register_handlers('gh', self.callback_func_for_gh)
        self.client.register_handlers('local_best_gain',
                                      self.callback_func_for_local_best_gain)
        self.client.register_handlers('split', self.callback_func_for_split)
        self.client.register_handlers('split_LR',
                                      self.callback_func_for_split_LR)

        self.client.order_feature(self.client.x)
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

    def compute_local_best_gain(self, tree_num, node_num, g, h):
        self.order_act_on_gh(g, h)
        best_gain = 0
        self.split_ref = {'feature_idx': None, 'value_idx': None}

        for feature_idx in range(self.client.my_num_of_feature):
            for value_idx in range(self.client.x.shape[0]):
                left_grad = np.sum(
                    self.ordered_list_g[feature_idx][:value_idx])
                right_grad = np.sum(
                    self.ordered_list_g[feature_idx]) - left_grad
                left_hess = np.sum(
                    self.ordered_list_h[feature_idx][:value_idx])
                right_hess = np.sum(
                    self.ordered_list_h[feature_idx]) - left_hess
                gain = self.client.cal_gain(left_grad, right_grad, left_hess,
                                            right_hess)

                if gain > best_gain:
                    best_gain = gain
                    self.split_ref['feature_idx'] = feature_idx
                    self.split_ref['value_idx'] = value_idx
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

    # client splits
    def callback_func_for_split(self, message: Message):
        tree_num, node_num = message.content
        feature_idx = self.split_ref['feature_idx']
        value_idx = self.split_ref['value_idx']
        self.client.feature_importance[feature_idx] += 1

        split_feature = self.client.feature_order[feature_idx]

        feature_value = self.client.x[:, feature_idx][
            self.client.feature_order[feature_idx][value_idx]]

        self.client.tree_list[tree_num][node_num].feature_idx = feature_idx
        self.client.tree_list[tree_num][node_num].feature_value = feature_value

        left_child_idx = np.zeros(len(self.client.x))
        for x in range(value_idx):
            left_child_idx[split_feature[x]] = 1
        right_child_idx = np.ones(len(self.client.x)) - left_child_idx

        self.client.comm_manager.send(
            Message(msg_type='split_LR',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.num_of_parties,
                    content=(tree_num, node_num, left_child_idx,
                             right_child_idx)))

    # label owner
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
