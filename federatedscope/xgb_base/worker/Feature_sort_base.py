import time

import numpy as np
import pandas as pd

from federatedscope.core.message import Message


class Feature_sort_base():
    def __init__(self, obj):
        self.client = obj
        self.total_feature_order_dict = dict()

    def _gain(self, grad, hess):
        return np.power(grad, 2) / (hess + self.client.lambda_)

    def cal_gain(self, left_grad, right_grad, left_hess, right_hess):
        left_gain = self._gain(left_grad, left_hess)
        right_gain = self._gain(right_grad, right_hess)
        total_gain = self._gain(left_grad + right_grad, left_hess + right_hess)
        return (left_gain + right_gain - total_gain) * 0.5 - self.client.gamma

    def preparation(self):
        self.client.register_handlers('feature_order',
                                      self.callback_func_for_feature_order)

        self.client.order_feature(self.client.x)
        # print(self.client.ID, self.client.feature_order)
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
            tree_num = 0
            self.total_feature_order_dict[self.client.ID -
                                          1] = self.client.feature_order
            self.total_feature_order_list = np.concatenate(
                list(self.total_feature_order_dict.values()))
            self.total_feature_order_dict = dict()
            self.compute_for_root(tree_num)

    # label owner
    def compute_for_root(self, tree_num):
        g, h = self.client.get_grad_and_hess(self.client.y, self.client.y_hat)
        node_num = 0
        self.client.tree_list[tree_num][node_num].grad = g
        self.client.tree_list[tree_num][node_num].hess = h
        self.client.tree_list[tree_num][node_num].indicator = np.ones(
            len(self.client.y))
        self.client.compute_for_node(tree_num, node_num)

    def perm_act_on_list(self, pi, list):
        res = np.zeros(len(list))
        for i in range(len(list)):
            res[i] = list[pi[i]]
        return res

    def order_act_on_gh(self, tree_num, node_num):
        self.client.total_ordered_g_list = [
            0
        ] * self.client.total_num_of_feature
        self.client.total_ordered_h_list = [
            0
        ] * self.client.total_num_of_feature
        for i in range(self.client.total_num_of_feature):
            self.client.total_ordered_g_list[i] = self.perm_act_on_list(
                self.total_feature_order_list[i],
                self.client.tree_list[tree_num][node_num].grad)
            self.client.total_ordered_h_list[i] = self.perm_act_on_list(
                self.total_feature_order_list[i],
                self.client.tree_list[tree_num][node_num].hess)
