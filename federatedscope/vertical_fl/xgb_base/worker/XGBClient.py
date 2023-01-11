import numpy as np
import logging

from federatedscope.core.workers import Client
from federatedscope.core.message import Message

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

        self.data = data
        self.own_label = ('y' in data['train'])
        self.msg_buffer = dict()
        self.client_num = self._cfg.federate.client_num
        self._init_data_related_var()
        # Add self-loop
        if self._cfg.federate.mode == 'distributed':
            self.comm_manager.add_neighbors(neighbor_id=self.ID,
                                            address={
                                                'host': self.comm_manager.host,
                                                'port': self.comm_manager.port
                                            })

        self.register_handlers('model_para', self.callback_func_for_model_para)
        self.register_handlers('data_sample',
                               self.callback_func_for_data_sample)
        self.register_handlers('feature_order',
                               self.callback_func_for_feature_order)
        self.register_handlers('finish', self.callback_func_for_finish)

    def train(self, tree_num, node_num=None):
        raise NotImplementedError

    def eval(self, tree_num):
        raise NotImplementedError

    def _init_data_related_var(self):

        self.feature_order = None
        self.merged_feature_order = None

        self.feature_partition = np.diff(self._cfg.vertical.dims, prepend=0)
        self.total_num_of_feature = self._cfg.vertical.dims[-1]
        self.num_of_feature = self.feature_partition[self.ID - 1]
        self.feature_importance = [0] * self.num_of_feature

        self.test_x = None
        self.test_y = None

    # all clients receive model para, and initial a tree list,
    # each contains self.num_of_trees trees
    # label-owner initials y_hat
    # label-owner sends "sample data" to others
    # label-owner calls self.preparation()

    def callback_func_for_model_para(self, message: Message):
        self.state = message.state

        if self.own_label:
            batch_index, self.feature_order = self.trainer.prepare_for_train()
            self.msg_buffer[self.ID] = self.feature_order
            receiver = [
                each for each in list(self.comm_manager.neighbors.keys())
                if each not in [self.ID, self.server_id]
            ]
            self.comm_manager.send(
                Message(msg_type='data_sample',
                        sender=self.ID,
                        state=self.state,
                        receiver=receiver,
                        content=batch_index))

    # other clients receive the data-sample information
    # other clients also call self.preparation()
    def callback_func_for_data_sample(self, message: Message):
        batch_index, sender = message.content, message.sender
        _, self.feature_order = self.trainer.prepare_for_train(
            index=batch_index)
        self.comm_manager.send(
            Message(msg_type='feature_order',
                    sender=self.ID,
                    state=self.state,
                    receiver=[sender],
                    content=self.feature_order))

    def callback_func_for_feature_order(self, message: Message):
        feature_order, sender = message.content, message.sender
        self.msg_buffer[sender] = feature_order
        self.check_and_move_on()

    def callback_func_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")
        self._monitor.finish_fl()

    def check_and_move_on(self):
        if len(self.msg_buffer) == self.client_num:
            self.merged_feature_order = np.concatenate([
                self.msg_buffer[idx] for idx in range(1, self.client_num + 1)
            ])
            self.train(tree_num=self.state)
