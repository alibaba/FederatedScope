import time

import numpy as np
import logging

from federatedscope.xgb_base.worker.Tree import Tree

from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.xgb_base.dataloader.utils import batch_iter

from federatedscope.xgb_base.worker.Feature_sort_base import Feature_sort_base
from federatedscope.xgb_base.worker.Test_base import Test_base

logger = logging.getLogger(__name__)


class XGBClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):

        super(XGBClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

        self.lambda_ = None
        self.gamma = None
        self.num_of_trees = None
        self.max_tree_depth = None

        self.data = data
        self.own_label = ('y' in self.data['train'])
        self.y_hat = None
        self.y = None
        self.num_of_parties = config.federate.client_num

        # self.num_of_feature_dict = {}

        self.dataloader = batch_iter(self.data['train'],
                                     self._cfg.data.batch_size,
                                     shuffled=True)

        # self.feature = None
        self.feature_order = None

        self.z = 0

        self.feature_list = [0] + config.xgb_base.dims
        self.feature_partition = [
            self.feature_list[i] - self.feature_list[i - 1]
            for i in range(1, len(self.feature_list))
        ]
        self.my_num_of_feature = self.feature_partition[self.ID - 1]
        self.total_num_of_feature = self.feature_list[-1]

        self.feature_order = [0] * self.my_num_of_feature

        self.total_ordered_g_list = [0] * self.total_num_of_feature
        self.total_ordered_h_list = [0] * self.total_num_of_feature

        # self.ss = AdditiveSecretSharing(shared_party_num=self.num_of_parties)
        # self.ns = Node_split()
        # self.fs = Feature_sort()
        self.fs = Feature_sort_base(self)
        self.ts = Test_base(self)

        self.register_handlers('model_para', self.callback_func_for_model_para)
        self.register_handlers('data_sample',
                               self.callback_func_for_data_sample)

        self.register_handlers('split', self.callback_func_for_split)

        self.register_handlers('children_index_vectors',
                               self.callback_func_for_children_index_vectors)

    # save the order of values in each feature
    def order_feature(self, data):
        for j in range(data.shape[1]):
            self.feature_order[j] = data[:, j].argsort()

    # define the function for computing grad and hess
    def get_grad_and_hess(self, y, pred):
        pred = np.array(pred)
        y = np.array(y)
        prob = 1.0 / (1.0 + np.exp(-pred))
        grad = prob - y
        hess = prob * (1.0 - prob)
        return grad, hess

    # sample data
    def sample_data(self, index=None):
        if index is None:
            assert self.own_label
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    # all clients receive model para, and initial a tree list,
    # each contains self.num_of_trees trees
    # label-owner initials y_hat
    # label-owner sends "sample data" to others
    # label-owner calls self.preparation()

    def callback_func_for_model_para(self, message: Message):
        self.lambda_, self.gamma, self.num_of_trees, self.max_tree_depth \
            = message.content
        self.tree_list = [
            Tree(self.max_tree_depth).tree for _ in range(self.num_of_trees)
        ]
        if self.own_label:
            self.batch_index, self.x, self.y = self.sample_data()
            # init y_hat
            self.y_hat = np.random.uniform(low=0.0, high=1.0, size=len(self.y))
            # self.y_hat = np.zeros(len(self.y))
            self.comm_manager.send(
                Message(
                    msg_type='data_sample',
                    sender=self.ID,
                    state=self.state,
                    receiver=[
                        each
                        for each in list(self.comm_manager.neighbors.keys())
                        if each != self.server_id
                    ],
                    content=self.batch_index))
            # self.preparation()
            self.fs.preparation()

    # other clients receive the data-sample information
    # other clients also call self.preparation()
    def callback_func_for_data_sample(self, message: Message):
        self.batch_index = message.content
        self.x = self.sample_data(index=self.batch_index)
        # self.preparation()
        self.fs.preparation()

    # label owner
    def compute_for_node(self, tree_num, node_num):
        if node_num >= 2**self.max_tree_depth - 1:
            self.prediction(tree_num)
        elif self.tree_list[tree_num][node_num].status == 'off':
            self.compute_for_node(tree_num, node_num + 1)
        elif node_num >= 2**(self.max_tree_depth - 1) - 1:
            self.set_weight(tree_num, node_num)
        else:
            self.fs.order_act_on_gh(tree_num, node_num)
            best_gain = 0
            split_ref = {'feature_idx': None, 'value_idx': None}

            for feature_idx in range(self.total_num_of_feature):
                for value_idx in range(self.x.shape[0]):
                    left_grad = np.sum(
                        self.total_ordered_g_list[feature_idx][:value_idx])
                    right_grad = np.sum(
                        self.total_ordered_g_list[feature_idx]) - left_grad
                    left_hess = np.sum(
                        self.total_ordered_h_list[feature_idx][:value_idx])
                    right_hess = np.sum(
                        self.total_ordered_h_list[feature_idx]) - left_hess
                    gain = self.fs.cal_gain(left_grad, right_grad, left_hess,
                                            right_hess)

                    if gain > best_gain:
                        best_gain = gain
                        split_ref['feature_idx'] = feature_idx
                        split_ref['value_idx'] = value_idx

            if best_gain > 0:
                print(best_gain, split_ref)
                for i in range(self.num_of_parties):
                    if self.feature_list[i] <= split_ref[
                            'feature_idx'] < self.feature_list[i + 1]:
                        self.tree_list[tree_num][node_num].member = i + 1
                        self.tree_list[tree_num][
                            node_num].feature_idx = split_ref[
                                'feature_idx'] - self.feature_list[i]
                        self.comm_manager.send(
                            Message(msg_type='split',
                                    sender=self.ID,
                                    state=self.state,
                                    receiver=i + 1,
                                    content=(tree_num, node_num, split_ref)))
                        break
            else:
                self.set_weight(tree_num, node_num)

    def split_for_lr(self, data, feature_value):
        left_index = [1 if x < feature_value else 0 for x in data]
        right_index = [1 if x >= feature_value else 0 for x in data]
        return left_index, right_index

    def callback_func_for_split(self, message: Message):
        tree_num, node_num, split_ref = message.content
        feature_idx = split_ref['feature_idx'] - self.feature_list[self.ID - 1]
        value_idx = split_ref['value_idx']
        # feature_value = sorted(self.x[:, feature_idx])[value_idx]
        feature_value = self.x[:, feature_idx][self.feature_order[feature_idx]
                                               [value_idx]]

        self.tree_list[tree_num][node_num].feature_idx = feature_idx
        self.tree_list[tree_num][node_num].feature_value = feature_value

        left_child_idx, right_child_idx = self.split_for_lr(
            self.x[:, feature_idx], feature_value)
        self.comm_manager.send(
            Message(msg_type='children_index_vectors',
                    sender=self.ID,
                    state=self.state,
                    receiver=self.num_of_parties,
                    content=(tree_num, node_num, left_child_idx,
                             right_child_idx)))

    # label owner
    def callback_func_for_children_index_vectors(self, message: Message):
        tree_num, node_num, left_child_idx, right_child_idx = message.content
        self.tree_list[tree_num][
            2 * node_num +
            1].grad = self.tree_list[tree_num][node_num].grad * left_child_idx
        self.tree_list[tree_num][
            2 * node_num +
            1].hess = self.tree_list[tree_num][node_num].hess * left_child_idx
        self.tree_list[tree_num][2 * node_num + 1].indicator = self.tree_list[
            tree_num][node_num].indicator * left_child_idx
        self.tree_list[tree_num][
            2 * node_num +
            2].grad = self.tree_list[tree_num][node_num].grad * right_child_idx
        self.tree_list[tree_num][
            2 * node_num +
            2].hess = self.tree_list[tree_num][node_num].hess * right_child_idx
        self.tree_list[tree_num][2 * node_num + 2].indicator = self.tree_list[
            tree_num][node_num].indicator * right_child_idx
        self.compute_for_node(tree_num, node_num + 1)

    def set_weight(self, tree_num, node_num):
        sum_of_g = np.sum(self.tree_list[tree_num][node_num].grad)
        sum_of_h = np.sum(self.tree_list[tree_num][node_num].hess)
        weight = -sum_of_g / (sum_of_h + self.lambda_)
        self.tree_list[tree_num][node_num].weight = weight
        print(tree_num, node_num, weight)
        self.tree_list[tree_num][node_num].status = 'off'
        tmp = [node_num]
        while tmp:
            x = tmp[0]
            self.tree_list[tree_num][x].status = 'off'
            tmp = tmp[1:]
            if 2 * x + 2 <= 2**self.max_tree_depth - 1:
                tmp.append(2 * x + 1)
                tmp.append(2 * x + 2)
        self.compute_for_node(tree_num, node_num + 1)

    def prediction(self, tree_num):
        node_num = 0
        self.compute_weight(tree_num, node_num)

    def compute_weight(self, tree_num, node_num):
        if node_num >= 2**(self.max_tree_depth) - 1:
            if tree_num == 0:
                self.y_hat = self.z
            else:
                self.y_hat += self.z
            self.z = 0
            yy = 1.0 / (1.0 + np.exp(-self.y_hat))
            # print(yy)
            yy[yy >= 0.5] = 1.
            yy[yy < 0.5] = 0
            acc = np.sum(yy == self.y) / len(self.y)
            print('Train accuracy: {:.2f}%'.format(acc * 100.0))

            if tree_num + 1 == self.num_of_trees:
                print("train over")

                self.comm_manager.send(
                    Message(msg_type='test',
                            sender=self.ID,
                            state=self.state,
                            receiver=self.server_id,
                            content=None))
            else:
                self.state += 1
                logger.info(
                    f'----------- Starting a new training round (Round '
                    f'#{self.state}) -------------')
                tree_num += 1
                # starts to build the next tree
                self.fs.compute_for_root(tree_num)
        else:
            if self.tree_list[tree_num][node_num].weight:
                self.z += self.tree_list[tree_num][
                    node_num].weight * self.tree_list[tree_num][
                        node_num].indicator
            self.compute_weight(tree_num, node_num + 1)
