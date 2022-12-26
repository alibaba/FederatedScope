import time

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
from federatedscope.core.message import Message


class Test_base:
    def __init__(self, obj):
        self.client = obj

        self.client.register_handlers(
            'split_lr_for_test_data',
            self.callback_func_for_split_lr_for_test_data)
        self.client.register_handlers('LR', self.callback_func_for_LR)

    def evaluation(self):

        loss = self.client.ls.loss(self.client.test_y, self.client.test_result)
        if self.client.criterion_type == 'CrossEntropyLoss':
            metric = self.client.ls.metric(self.client.test_y,
                                           self.client.test_result)

            metrics = {
                'test_loss': loss,
                'test_acc': metric[1],
                'test_total': len(self.client.test_y)
            }
        else:
            metrics = {
                'test_loss': loss,
                'test_total': len(self.client.test_y)
            }
        return metrics

    def test_for_root(self, tree_num):
        node_num = 0
        self.client.tree_list[tree_num][node_num].indicator = np.ones(
            self.client.test_x.shape[0])
        self.test_for_node(tree_num, node_num)

    def test_for_node(self, tree_num, node_num):
        if node_num >= 2**self.client.max_tree_depth - 1:
            if tree_num + 1 < self.client.num_of_trees:

                # TODO: add feedback during training
                logger.info(f'----------- Building a new tree (Tree '
                            f'#{tree_num + 1}) -------------')
                # build the next tree
                self.client.fs.compute_for_root(tree_num + 1)

            else:
                metrics = self.evaluation()
                self.client.comm_manager.send(
                    Message(msg_type='test_result',
                            sender=self.client.ID,
                            state=self.client.state,
                            receiver=self.client.server_id,
                            content=(tree_num, metrics)))

                self.client.comm_manager.send(
                    Message(msg_type='send_feature_importance',
                            sender=self.client.ID,
                            state=self.client.state,
                            receiver=[
                                each for each in list(
                                    self.client.comm_manager.neighbors.keys())
                                if each != self.client.server_id
                                and each != self.client.ID
                            ],
                            content='None'))
                self.client.comm_manager.send(
                    Message(msg_type='feature_importance',
                            sender=self.client.ID,
                            state=self.client.state,
                            receiver=self.client.server_id,
                            content=self.client.feature_importance))
        elif self.client.tree_list[tree_num][node_num].weight:
            self.client.test_result += self.client.tree_list[tree_num][
                node_num].indicator * self.client.tree_list[tree_num][
                    node_num].weight * self.client.eta
            self.test_for_node(tree_num, node_num + 1)
        elif self.client.tree_list[tree_num][node_num].member:
            self.client.comm_manager.send(
                Message(
                    msg_type='split_lr_for_test_data',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.tree_list[tree_num][node_num].member,
                    content=(tree_num, node_num)))
        else:
            self.test_for_node(tree_num, node_num + 1)

    def callback_func_for_split_lr_for_test_data(self, message: Message):
        tree_num, node_num = message.content
        feature_idx = self.client.tree_list[tree_num][node_num].feature_idx
        feature_value = self.client.tree_list[tree_num][node_num].feature_value
        L, R = self.client.split_for_lr(self.client.test_x[:, feature_idx],
                                        feature_value)
        self.client.comm_manager.send(
            Message(msg_type='LR',
                    sender=self.client.ID,
                    state=self.client.state,
                    receiver=self.client.num_of_parties,
                    content=(tree_num, node_num, L, R)))

    def callback_func_for_LR(self, message: Message):
        tree_num, node_num, L, R = message.content
        self.client.tree_list[tree_num][2 * node_num +
                                        1].indicator = self.client.tree_list[
                                            tree_num][node_num].indicator * L
        self.client.tree_list[tree_num][2 * node_num +
                                        2].indicator = self.client.tree_list[
                                            tree_num][node_num].indicator * R
        self.test_for_node(tree_num, node_num + 1)
