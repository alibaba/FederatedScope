import time

import numpy as np
import logging

from federatedscope.vertical_fl.xgb_base.worker.Tree import Tree

from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.vertical_fl.dataloader.utils import batch_iter

from federatedscope.vertical_fl.xgb_base.worker.Feature_sort_base\
    import Feature_sort_base
from federatedscope.vertical_fl.xgb_base.worker.Feature_sort_by_bin\
    import Feature_sort_by_bin
from federatedscope.vertical_fl.xgb_base.worker.Test_base import Test_base

from federatedscope.vertical_fl.xgb_base.worker.Loss_function \
    import TwoClassificationloss, Regression_by_mseloss, Regression_by_maeloss

logger = logging.getLogger(__name__)


class XGBClient(Client):
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

        super(XGBClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

        self.lambda_ = None
        self.gamma = None
        self.num_of_trees = None
        self.max_tree_depth = None
        self.vertical_dims = self._cfg.vertical_dims

        self.federate_mode = config.federate.mode

        self.bin_num = config.train.optimizer.bin_num
        self.batch_size = config.dataloader.batch_size

        self.data = data
        self.own_label = ('y' in self.data['train'])
        self._init_data_related_var()

        self.register_handlers('model_para', self.callback_func_for_model_para)
        self.register_handlers('data_sample',
                               self.callback_func_for_data_sample)
        self.register_handlers('compute_next_node',
                               self.callback_func_for_compute_next_node)
        self.register_handlers('send_feature_importance',
                               self.callback_func_for_send_feature_importance)
        self.register_handlers('finish', self.callback_func_for_finish)

    def _init_data_related_var(self):
        self.test_x = self.data['test']['x']
        if self.own_label:
            self.test_y = self.data['test']['y']

        self.test_result = np.zeros(self.test_x.shape[0])

        self.y_hat = None
        self.y = None
        self.num_of_parties = self._cfg.federate.client_num

        self.dataloader = batch_iter(self.data['train'],
                                     self.batch_size,
                                     shuffled=True)

        self.feature_order = None

        self.z = 0

        self.feature_list = [0] + self.vertical_dims
        self.feature_partition = [
            self.feature_list[i] - self.feature_list[i - 1]
            for i in range(1, len(self.feature_list))
        ]
        self.my_num_of_feature = self.feature_partition[self.ID - 1]
        self.total_num_of_feature = self.feature_list[-1]

        self.feature_order = [0] * self.my_num_of_feature

        self.feature_importance = [0] * self.my_num_of_feature
        # self.fs = Feature_sort()
        # the following two lines are the two alogs, where
        #   the first one corresponding to sending the whole feature order
        #   the second one corresponding to sending the bins of feature order
        self.opboost_noise_use = self._cfg.opboost_noise.use
        if self._cfg.xgb_base.use_bin:
            self.fs = Feature_sort_by_bin(self, bin_num=self.bin_num)
        elif self.opboost_noise_use:
            self.opboost_algo = self._cfg.opboost_noise.algo
            self.epsilon = self._cfg.opboost_noise.epsilon
            self.epsilon_prt = self._cfg.opboost_noise.epsilon_prt
            self.epsilon_ner = self._cfg.opboost_noise.epsilon_ner
            self.opboost_partition_num = self._cfg.opboost_noise.partition_num
            self.fs = Feature_sort_base(self,
                                        epsilon=self.epsilon,
                                        epsilon_prt=self.epsilon_prt,
                                        epsilon_ner=self.epsilon_ner)
        else:
            self.fs = Feature_sort_base(self)

        self.ts = Test_base(self)

        self.criterion_type = self._cfg.criterion.type
        if self.criterion_type == 'CrossEntropyLoss':
            self.ls = TwoClassificationloss()
        elif self.criterion_type == 'Regression':
            self.ls = Regression_by_mseloss()

    # save the order of values in each feature
    def order_feature(self, data):
        for j in range(data.shape[1]):
            self.feature_order[j] = data[:, j].argsort()

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

        # client adds his own ID and address in his comm_manager.neighbors
        # to send and receive messages from himself
        if self.federate_mode == 'distributed':
            self.comm_manager.add_neighbors(neighbor_id=self.ID,
                                            address={
                                                'host': self.comm_manager.host,
                                                'port': self.comm_manager.port
                                            })
        self.tree_list = [
            Tree(self.max_tree_depth).tree for _ in range(self.num_of_trees)
        ]
        if self.own_label:
            self.batch_index, self.x, self.y = self.sample_data()
            # init y_hat
            self.y_hat = np.random.uniform(low=0.0, high=1.0, size=len(self.y))
            logger.info(f'---------- Building a new tree (Tree '
                        f'#{self.state}) -------------')
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

    # other clients receive the data-sample information
    # other clients also call self.preparation()
    def callback_func_for_data_sample(self, message: Message):
        self.batch_index = message.content
        self.x = self.sample_data(index=self.batch_index)
        self.fs.preparation()

    def _gain(self, grad, hess):
        return np.power(grad, 2) / (hess + self.lambda_)

    def cal_gain(self, left_grad, right_grad, left_hess, right_hess):
        left_gain = self._gain(left_grad, left_hess)
        right_gain = self._gain(right_grad, right_hess)
        total_gain = self._gain(left_grad + right_grad, left_hess + right_hess)
        return (left_gain + right_gain - total_gain) * 0.5 - self.gamma

    def split_for_lr(self, data, feature_value):
        left_index = [1 if x < feature_value else 0 for x in data]
        right_index = [1 if x >= feature_value else 0 for x in data]
        return left_index, right_index

    # label owner
    def callback_func_for_compute_next_node(self, message: Message):
        tree_num, node_num = message.content
        self.fs.compute_for_node(tree_num, node_num + 1)

    def set_weight(self, tree_num, node_num):
        sum_of_g = np.sum(self.tree_list[tree_num][node_num].grad)
        sum_of_h = np.sum(self.tree_list[tree_num][node_num].hess)
        weight = -sum_of_g / (sum_of_h + self.lambda_)
        self.tree_list[tree_num][node_num].weight = weight
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

    def callback_func_for_finish(self, message: Message):
        pass
