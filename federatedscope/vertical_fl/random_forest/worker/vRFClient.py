from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.vertical_fl.xgb_base.worker.Tree import Tree
from federatedscope.vertical_fl.random_forest.worker.Feature_sort_by_bin\
    import Feature_sort_by_bin
from federatedscope.vertical_fl.random_forest.worker.Feature_sort_base\
    import Feature_sort_base
from federatedscope.vertical_fl.dataloader.utils import batch_iter
from federatedscope.vertical_fl.random_forest.worker.Test_base import Test_base
from federatedscope.vertical_fl.random_forest.worker.Loss_function\
    import TwoClassificationloss, Regression_by_mseloss

from federatedscope.vertical_fl.Paillier import abstract_paillier

import numpy as np
import logging

logger = logging.getLogger(__name__)


class vRFClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(vRFClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

        self.vertical_dims = config.random_forest.dims

        self.bin_num = config.train.optimizer.bin_num
        self.batch_size = config.data.batch_size
        self.data = data
        self.num_of_trees = None
        self.max_tree_depth = None
        self.own_label = ('y' in self.data['train'])
        self.criterion_type = config.criterion.type
        self._init_data_related_var()

        self.register_handlers('model_para', self.callback_func_for_model_para)
        self.register_handlers('data_sample',
                               self.callback_func_for_data_sample)
        self.register_handlers('compute_next_node',
                               self.callback_func_for_compute_next_node)
        self.register_handlers('send_feature_importance',
                               self.callback_func_for_send_feature_importance)

    def _init_data_related_var(self):
        self.z = 0
        self.test_x = self.data['test']['x']
        self.test_result = np.zeros(self.test_x.shape[0])
        if self.own_label:
            self.test_y = self.data['test']['y']
        self.num_of_parties = self._cfg.federate.client_num
        self.dataloader = batch_iter(self.data['train'],
                                     self._cfg.data.batch_size,
                                     shuffled=True)
        self.feature_list = [0] + self.vertical_dims
        self.feature_partition = [
            self.feature_list[i] - self.feature_list[i - 1]
            for i in range(1, len(self.feature_list))
        ]
        self.my_num_of_feature = self.feature_partition[self.ID - 1]
        self.total_num_of_feature = self.feature_list[-1]
        self.feature_order = [0] * self.my_num_of_feature
        self.feature_importance = [0] * self.my_num_of_feature
        self.z = 0

        # self.fs = Feature_sort_by_bin(self, bin_num=self.bin_num)
        self.fs = Feature_sort_base(self)
        self.ts = Test_base(self)
        if self.criterion_type == 'CrossEntropyLoss':
            self.ls = TwoClassificationloss()
        elif self.criterion_type == 'Regression':
            self.ls = Regression_by_mseloss()

    def order_feature(self, data):
        for j in range(data.shape[1]):
            self.feature_order[j] = data[:, j].argsort()

    def sample_data(self, index=None):
        if index is None:
            assert self.own_label
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    def callback_func_for_model_para(self, message: Message):
        self.num_of_trees, self.max_tree_depth = message.content
        self.tree_list = [
            Tree(self.max_tree_depth).tree for _ in range(self.num_of_trees)
        ]
        self.start_new_train_round()

    def start_new_train_round(self):
        if self.own_label:
            logger.info(f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
            self.batch_index, self.x, self.y = self.sample_data()
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
            self.fs.preparation()

    def callback_func_for_data_sample(self, message: Message):
        self.batch_index = message.content
        self.x = self.sample_data(index=self.batch_index)
        self.fs.preparation()

    # label owner
    def callback_func_for_compute_next_node(self, message: Message):
        tree_num, node_num = message.content
        self.fs.compute_for_node(tree_num, node_num + 1)

    def prediction(self, tree_num):
        node_num = 0
        self.compute_weight(tree_num, node_num)

    def compute_weight(self, tree_num, node_num):
        if node_num >= 2**self.max_tree_depth - 1:
            if tree_num == 0:
                self.y_hat = self.z
            else:
                self.y_hat += self.z
            self.z = 0
            self.ts.test_for_root(tree_num)
        else:
            if self.tree_list[tree_num][node_num].weight:
                self.z += self.tree_list[tree_num][
                    node_num].weight * self.tree_list[tree_num][
                        node_num].indicator
            self.compute_weight(tree_num, node_num + 1)

    def callback_func_for_send_feature_importance(self, message: Message):
        self.comm_manager.send(
            Message(msg_type='feature_importance',
                    sender=self.ID,
                    state=self.state,
                    receiver=self.server_id,
                    content=self.feature_importance))

    def split_for_lr(self, data, feature_value):
        left_index = [1 if x < feature_value else 0 for x in data]
        right_index = [1 if x >= feature_value else 0 for x in data]
        return left_index, right_index

    def _gini(self, iv, y):
        total_num = sum(iv)
        positive_num = np.dot(iv, y)
        negative_num = total_num - positive_num
        return 1 - (positive_num / total_num)**2 - (negative_num /
                                                    total_num)**2

    def cal_gini(self, left, iv, y):
        left = left * iv
        right = [iv[i] - left[i] for i in range(len(left))]
        left_y = left * y
        right_y = right * y
        if sum(left_y) == 0 or sum(right_y) == 0:
            return float('inf')
        left_gini = self._gini(left, left_y)
        right_gini = self._gini(right, right_y)
        total_num = sum(iv)
        return sum(left) / total_num * left_gini + sum(
            right) / total_num * right_gini

    def set_weight(self, tree_num, node_num):
        real_y = []
        for i in range(len(self.tree_list[tree_num][node_num].indicator)):
            if self.tree_list[tree_num][node_num].indicator[i] != 0:
                real_y.append(self.tree_list[tree_num][node_num].label[i])
        if self.criterion_type == 'CrossEntropyLoss':
            positive_num = sum(real_y)
            if positive_num >= len(real_y) / 2:
                self.tree_list[tree_num][node_num].weight = 1
            else:
                self.tree_list[tree_num][node_num].weight = 0
        elif self.criterion_type == 'Regression':
            self.tree_list[tree_num][node_num].weight = np.mean(real_y)
        self.tree_list[tree_num][node_num].status = 'off'
        tmp = [node_num]
        while tmp:
            x = tmp[0]
            self.tree_list[tree_num][x].status = 'off'
            tmp = tmp[1:]
            if 2 * x + 2 <= 2**self.max_tree_depth - 1:
                tmp.append(2 * x + 1)
                tmp.append(2 * x + 2)
        self.fs.compute_for_node(tree_num, node_num + 1)
