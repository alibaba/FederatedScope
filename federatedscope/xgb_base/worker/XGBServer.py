from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.xgb_base.worker.Tree import Tree

import numpy as np
import logging

logger = logging.getLogger(__name__)


class XGBServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=2,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(XGBServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        self.lambda_ = config.train.optimizer.lambda_
        self.gamma = config.train.optimizer.gamma
        self.num_of_trees = config.train.optimizer.num_of_trees
        self.max_tree_depth = config.train.optimizer.max_tree_depth

        self.num_of_parties = config.federate.client_num

        self.batch_size = config.data.batch_size

        self.feature_partition = [0] + config.xgb_base.dims
        self.feature_partition = [
            self.feature_partition[i + 1] - self.feature_partition[i]
            for i in range(len(self.feature_partition) - 1)
        ]
        self.total_num_of_feature = config.xgb_base.dims[-1]
        self.feature_list = [0] + config.xgb_base.dims

        self.data = data

        self.tree_list = [
            Tree(self.max_tree_depth).tree for _ in range(self.num_of_trees)
        ]

        self.register_handlers('test', self.callback_func_for_test)

    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_client_address()
            self.broadcast_model_para()

    def broadcast_model_para(self):
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    state=self.state,
                    content=(self.lambda_, self.gamma, self.num_of_trees,
                             self.max_tree_depth)))

    def callback_func_for_test(self, message: Message):
        test_x = self.data['test']['x']
        test_y = self.data['test']['y']
        for i in range(self.num_of_parties):
            test_data = test_x[:,
                               self.feature_list[i]:self.feature_list[i + 1]]
            self.comm_manager.send(
                Message(msg_type='test_data',
                        sender=self.ID,
                        receiver=i + 1,
                        state=self.state,
                        content=test_data))
        self.comm_manager.send(
            Message(msg_type='test_value',
                    sender=self.ID,
                    receiver=self.num_of_parties,
                    state=self.state,
                    content=test_y))
