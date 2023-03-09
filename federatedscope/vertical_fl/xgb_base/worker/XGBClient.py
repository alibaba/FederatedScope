import numpy as np
import logging
import copy

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
        self.msg_buffer = {'train': {}, 'eval': {}}
        self.client_num = self._cfg.federate.client_num

        self.feature_order = None
        self.merged_feature_order = None

        self.feature_partition = np.diff(self._cfg.vertical.dims, prepend=0)
        self.total_num_of_feature = self._cfg.vertical.dims[-1]
        self.num_of_feature = self.feature_partition[self.ID - 1]
        self.feature_importance = [0] * self.num_of_feature

        self._init_data_related_var()

        self.register_handlers('model_para', self.callback_func_for_model_para)
        self.register_handlers('data_sample',
                               self.callback_func_for_data_sample)
        self.register_handlers('training_info',
                               self.callback_func_for_training_info)
        self.register_handlers('finish', self.callback_func_for_finish)

    def train(self, tree_num, node_num=None, training_info=None):
        raise NotImplementedError

    def eval(self, tree_num):
        raise NotImplementedError

    def _init_data_related_var(self):

        self.trainer._init_for_train()
        self.test_x = None
        self.test_y = None

    # all clients receive model para, and initial a tree list,
    # each contains self.num_of_trees trees
    # label-owner initials y_hat
    # label-owner sends "sample data" to others
    def callback_func_for_model_para(self, message: Message):
        self.state = message.state

        self.trainer.prepare_for_train()
        if self.own_label:
            batch_index, feature_order_info = self.trainer.fetch_train_data()
            self.start_a_new_training_round(batch_index,
                                            feature_order_info,
                                            tree_num=0)

    # other clients receive the data-sample information
    def callback_func_for_data_sample(self, message: Message):
        batch_index, sender = message.content, message.sender
        _, feature_order_info = self.trainer.fetch_train_data(
            index=batch_index)
        self.feature_order = feature_order_info['feature_order']

        if self._cfg.vertical.mode == 'order_based':
            training_info = feature_order_info
        elif self._cfg.vertical.mode == 'label_based':
            training_info = 'dummy_info'
        else:
            raise TypeError(f'The expected types of vertical.mode include '
                            f'["label_based", "order_based"], but got '
                            f'{self._cfg.vertical.mode}.')

        self.comm_manager.send(
            Message(msg_type='training_info',
                    sender=self.ID,
                    state=self.state,
                    receiver=[sender],
                    content=training_info))

    def callback_func_for_training_info(self, message: Message):
        feature_order_info, sender = message.content, message.sender
        self.msg_buffer['train'][sender] = feature_order_info
        self.check_and_move_on()

    def callback_func_for_finish(self, message: Message):
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")
        # self._monitor.finish_fl()

    def start_a_new_training_round(self,
                                   batch_index,
                                   feature_order_info,
                                   tree_num=0):
        self.msg_buffer['train'].clear()
        self.feature_order = feature_order_info['feature_order']
        self.msg_buffer['train'][self.ID] = feature_order_info \
            if self._cfg.vertical.mode == 'order_based' else 'dummy_info'
        self.state = tree_num
        receiver = [
            each for each in list(self.comm_manager.neighbors.keys())
            if each != self.server_id
        ]
        send_message = Message(msg_type='data_sample',
                               sender=self.ID,
                               state=self.state,
                               receiver=receiver,
                               content=batch_index)
        self.comm_manager.send(send_message)

    def check_and_move_on(self):
        if len(self.msg_buffer['train']) == self.client_num:
            received_training_infos = copy.deepcopy(self.msg_buffer['train'])
            self.msg_buffer['train'].clear()
            self.train(tree_num=self.state,
                       training_info=received_training_infos)
