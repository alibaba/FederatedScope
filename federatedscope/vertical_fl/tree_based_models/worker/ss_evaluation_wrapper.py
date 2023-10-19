import types
import logging
from abc import abstractmethod

import numpy as np

from federatedscope.vertical_fl.loss.utils import get_vertical_loss
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_ss_evaluation(client):
    def eval(self, tree_num):
        self.pe_dict = dict()
        self.pf_dict = dict()
        self.plf_dict = dict()
        self.prf_dict = dict()

        self.criterion = get_vertical_loss(loss_type=self._cfg.criterion.type,
                                           model_type=self._cfg.model.type)
        if self.test_x is None:
            self.test_x, self.test_y = self._fetch_test_data()
            self.merged_test_result = list()
        self.test_result_dict = dict()
        self.test_result_piece_list = list()

        indicator_piece_list = self.ss.secret_split(np.ones(
            self.test_x.shape[0]),
                                                    cls='ss_piece')

        self.model[tree_num][0].indicator = indicator_piece_list[-1]
        self.tree_num = tree_num
        self.node_num = 0
        for i in range(self.client_num - 1):
            self.comm_manager.send(
                Message(msg_type='indicator_piece',
                        sender=self.ID,
                        state=self.state,
                        receiver=[i + 1],
                        content=(tree_num, 0, indicator_piece_list[i])))
        self._test_for_node(tree_num, node_num=0)

    def _fetch_test_data(self):
        test_x = self.data['test']['x']
        test_y = self.data['test']['y'] if 'y' in self.data['test'] else None

        return test_x, test_y

    def callback_func_for_indicator_piece(self, message: Message):
        tree_num, node_num, indicator_piece = message.content
        self.tree_num = tree_num
        self.node_num = node_num
        self.model[tree_num][node_num].indicator = indicator_piece
        self.test_result_piece_list = list()

    def _feedback_eval_metrics(self):
        test_loss = self.criterion.get_loss(self.test_y,
                                            self.merged_test_result)
        metrics = self.criterion.get_metric(self.test_y,
                                            self.merged_test_result)
        modified_metrics = dict()
        for key in metrics.keys():
            if 'test' not in key:
                modified_metrics['test_' + key] = metrics[key]
            else:
                modified_metrics[key] = metrics[key]
        modified_metrics.update({
            'test_loss': test_loss,
            'test_total': len(self.test_y)
        })

        self.comm_manager.send(
            Message(msg_type='eval_metric',
                    sender=self.ID,
                    state=self.state,
                    receiver=[self.server_id],
                    content=modified_metrics))
        self.comm_manager.send(
            Message(msg_type='feature_importance',
                    sender=self.ID,
                    state=self.state,
                    receiver=[self.server_id],
                    content=self.feature_importance))
        self.comm_manager.send(
            Message(msg_type='ask_for_feature_importance',
                    sender=self.ID,
                    state=self.state,
                    receiver=[
                        each
                        for each in list(self.comm_manager.neighbors.keys())
                        if each != self.server_id
                    ],
                    content='None'))

    def _test_for_node(self, tree_num, node_num):
        # All nodes have been traversed
        if node_num >= 2**self.model.max_depth - 1:
            part_result = self.ss.secret_add_lists(self.test_result_piece_list)
            self.test_result_dict[self.ID] = part_result
            self.comm_manager.send(
                Message(
                    msg_type='ask_for_part_result',
                    sender=self.ID,
                    state=self.state,
                    receiver=[
                        each
                        for each in list(self.comm_manager.neighbors.keys())
                        if each != self.server_id
                    ],
                    content=(tree_num, node_num)))
        # The client owns the weight
        elif self.model[tree_num][node_num].weight:
            if self._cfg.model.type in ['xgb_tree', 'gbdt_tree']:
                eta = self._cfg.train.optimizer.eta
            else:
                eta = 1.0

            weight = self.model[tree_num][node_num].weight * eta
            weight_piece_list = self.ss.secret_split(weight)
            weight_piece_list = weight_piece_list
            self.model[tree_num][node_num].weight_piece = weight_piece_list[-1]
            for i in range(self.client_num - 1):
                self.comm_manager.send(
                    Message(msg_type='weight_piece',
                            sender=self.ID,
                            state=self.state,
                            receiver=[i + 1],
                            content=(tree_num, node_num,
                                     weight_piece_list[i])))

            self.ss_multiplicative(self.model[tree_num][node_num].weight_piece,
                                   self.model[tree_num][node_num].indicator,
                                   self.client_num, 'weight')
        # Other client owns the weight, need to communicate
        elif self.model[tree_num][node_num].member:
            send_message = Message(
                msg_type='split_request',
                sender=self.ID,
                state=self.state,
                receiver=[self.model[tree_num][node_num].member],
                content=(tree_num, node_num))
            if self.model[tree_num][node_num].member == self.ID:
                self.callback_func_for_split_request(send_message)
            else:
                self.comm_manager.send(send_message)

        else:
            self._test_for_node(tree_num, node_num + 1)

    def callback_func_for_ask_for_part_result(self, message: Message):
        tree_num, node_num = message.content
        sender = message.sender
        part_result = self.ss.secret_add_lists(self.test_result_piece_list)
        self.comm_manager.send(
            Message(msg_type='part_result',
                    sender=self.ID,
                    state=self.state,
                    receiver=[sender],
                    content=(tree_num, node_num, part_result)))

    def callback_func_for_part_result(self, message: Message):
        tree_num, node_num, part_result = message.content
        self.test_result_dict[message.sender] = part_result
        if len(self.test_result_dict) == self.client_num:
            result = self.ss.secret_reconstruct(
                list(self.test_result_dict.values()))
            self.merged_test_result.append(result)
            if (
                    tree_num + 1
            ) % self._cfg.eval.freq == 0 or \
                    tree_num + 1 == self._cfg.model.num_of_trees:
                self._feedback_eval_metrics()
            self.eval_finish_flag = True
            self._check_eval_finish(tree_num)

    def callback_func_for_weight_piece(self, message: Message):
        tree_num, node_num, weight_piece = message.content
        self.model[tree_num][node_num].weight_piece = weight_piece
        self.ss_multiplicative(self.model[tree_num][node_num].weight_piece,
                               self.model[tree_num][node_num].indicator,
                               self.client_num, 'weight')

    def set_weight(self):
        self.test_result_piece_list.append(self.res)
        self.node_num += 1
        if self.own_label:
            self._test_for_node(self.tree_num, self.node_num)

    def callback_func_for_split_request(self, message: Message):
        if self.test_x is None:
            self.test_x, self.test_y = self._fetch_test_data()
        tree_num, node_num = message.content
        feature_idx = self.model[tree_num][node_num].feature_idx
        feature_value = self.model[tree_num][node_num].feature_value
        left_child, right_child = self.model[tree_num].split_childern(
            self.test_x[:, feature_idx], feature_value)

        left_child_piece_list = self.ss.secret_split(left_child,
                                                     cls='ss_piece')
        right_child_piece_list = self.ss.secret_split(right_child,
                                                      cls='ss_piece')
        self.left_child_piece = left_child_piece_list[self.ID - 1]
        self.right_child_piece = right_child_piece_list[self.ID - 1]
        for i in range(self.client_num):
            if i + 1 != self.ID:
                self.comm_manager.send(
                    Message(msg_type='split_result',
                            sender=self.ID,
                            state=self.state,
                            receiver=[i + 1],
                            content=(tree_num, node_num,
                                     left_child_piece_list[i],
                                     right_child_piece_list[i])))
        self.ss_multiplicative(self.model[tree_num][node_num].indicator,
                               self.left_child_piece, self.client_num,
                               'left_child')

    @abstractmethod
    def ss_multiplicative(self,
                          secret1,
                          secret2,
                          shared_party_num,
                          behavior=None):
        pass

    def callback_func_for_split_result(self, message: Message):
        tree_num, node_num, left_child_piece, right_child_piece \
            = message.content
        self.left_child_piece = left_child_piece
        self.right_child_piece = right_child_piece
        self.ss_multiplicative(self.model[tree_num][node_num].indicator,
                               self.left_child_piece, self.client_num,
                               'left_child')

    def set_left_child(self):
        self.model[self.tree_num][2 * self.node_num + 1].indicator = self.res

        self.ss_multiplicative(
            self.model[self.tree_num][self.node_num].indicator,
            self.right_child_piece, self.client_num, 'right_child')

    def set_right_child(self):
        self.model[self.tree_num][2 * self.node_num + 2].indicator = self.res
        self.node_num += 1
        if self.own_label:
            self._test_for_node(self.tree_num, self.node_num)

    def callback_func_for_feature_importance(self, message: Message):
        state = message.state
        self.comm_manager.send(
            Message(msg_type='feature_importance',
                    sender=self.ID,
                    state=state,
                    receiver=[self.server_id],
                    content=self.feature_importance))

    # Bind method to instance
    client.eval = types.MethodType(eval, client)
    client._fetch_test_data = types.MethodType(_fetch_test_data, client)
    client._test_for_node = types.MethodType(_test_for_node, client)
    client._feedback_eval_metrics = types.MethodType(_feedback_eval_metrics,
                                                     client)

    # client.ss_multiplicative = types.MethodType(ss_multiplicative, client)
    client.set_left_child = types.MethodType(set_left_child, client)
    client.set_right_child = types.MethodType(set_right_child, client)
    client.set_weight = types.MethodType(set_weight, client)

    client.callback_func_for_indicator_piece = types.MethodType(
        callback_func_for_indicator_piece, client)
    client.callback_func_for_weight_piece = types.MethodType(
        callback_func_for_weight_piece, client)
    client.callback_func_for_ask_for_part_result = types.MethodType(
        callback_func_for_ask_for_part_result, client)
    client.callback_func_for_part_result = types.MethodType(
        callback_func_for_part_result, client)

    client.callback_func_for_split_request = types.MethodType(
        callback_func_for_split_request, client)
    client.callback_func_for_split_result = types.MethodType(
        callback_func_for_split_result, client)
    client.callback_func_for_feature_importance = types.MethodType(
        callback_func_for_feature_importance, client)

    # Register handler functions
    client.register_handlers('split_request',
                             client.callback_func_for_split_request)
    client.register_handlers('split_result',
                             client.callback_func_for_split_result)
    client.register_handlers('ask_for_feature_importance',
                             client.callback_func_for_feature_importance)

    client.register_handlers('indicator_piece',
                             client.callback_func_for_indicator_piece)
    client.register_handlers('weight_piece',
                             client.callback_func_for_weight_piece)
    client.register_handlers('ask_for_part_result',
                             client.callback_func_for_ask_for_part_result)
    client.register_handlers('part_result',
                             client.callback_func_for_part_result)
    return client


def wrap_server_for_ss_evaluation(server):
    return server
