import types
import logging
import copy

from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_train(client):
    def train(self, tree_num, node_num=None, training_info=None):
        if node_num is None:
            logger.info(f'----------- Building a new tree (Tree '
                        f'#{tree_num}) -------------')
            train_flag, results = self.trainer.train(
                training_info=training_info, tree_num=tree_num)
        else:
            assert node_num is not None
            train_flag, results = self.trainer.train(tree_num=tree_num,
                                                     node_num=node_num)

        if train_flag == 'train_finish':
            # Report evaluation results
            self.eval_finish_flag = False
            self.eval(tree_num)
            self._check_eval_finish(tree_num)
        elif train_flag == 'call_for_node_split':
            split_ref, tree_num, node_num = results
            self._find_and_send_split(split_ref, tree_num, node_num)
        elif train_flag == 'call_for_local_gain':
            g, h, indicator, tree_num, node_num = results
            send_message = Message(
                msg_type='grad_and_hess',
                sender=self.ID,
                state=self.state,
                receiver=[
                    each for each in list(self.comm_manager.neighbors.keys())
                    if each != self.server_id
                ],
                content=(tree_num, node_num, g, h, indicator))
            self.comm_manager.send(send_message)
            # imitate send this message to itself
            self.callback_funcs_for_grad_and_hess(send_message)
        else:
            raise ValueError(f'The handler of {train_flag} is not defined.')

    def _check_eval_finish(self, tree_num):
        if self.eval_finish_flag:
            self.eval_finish_flag = False
            if tree_num + 1 < self._cfg.model.num_of_trees:
                batch_index, feature_order_info = \
                    self.trainer.fetch_train_data()
                self.start_a_new_training_round(batch_index,
                                                feature_order_info,
                                                tree_num=tree_num + 1)

    def _find_and_send_split(self, split_ref, tree_num, node_num):
        accum_dim = 0
        for index, dim in enumerate(self.trainer.client_feature_num):
            if split_ref['feature_idx'] < accum_dim + dim:
                client_id = index + 1
                self.model[tree_num][node_num].member = client_id

                split_ref['feature_idx'] -= accum_dim
                split_child = False
                send_message = Message(msg_type='split',
                                       sender=self.ID,
                                       state=self.state,
                                       receiver=[client_id],
                                       content=(tree_num, node_num, split_ref,
                                                split_child))
                if client_id == self.ID:
                    self.callback_func_for_split(send_message)
                else:
                    self.comm_manager.send(send_message)

                break
            else:
                accum_dim += dim

    def callback_func_for_split(self, message: Message):
        tree_num, node_num, split_ref, split_child = message.content

        if split_ref is None:
            feature_idx = self.trainer.split_ref['feature_idx']
            value_idx = self.trainer.split_ref['value_idx']
        else:
            feature_idx = split_ref['feature_idx']
            value_idx = split_ref['value_idx']

        sender = message.sender
        self.state = message.state
        abs_feature_idx = self.trainer.get_abs_feature_idx(feature_idx)
        self.feature_importance[abs_feature_idx] += 1
        if hasattr(self.trainer, 'get_abs_value_idx'):
            abs_value_idx = self.trainer.get_abs_value_idx(
                feature_idx, value_idx)
        else:
            abs_value_idx = value_idx

        feature_value = self.trainer.get_feature_value(feature_idx,
                                                       abs_value_idx)

        self.model[tree_num][node_num].feature_idx = abs_feature_idx
        self.model[tree_num][node_num].feature_value = feature_value

        if split_child:
            split_feature = self.trainer.client_feature_order[feature_idx]
            left_child, right_child = self.trainer.get_children_indicator(
                value_idx=abs_value_idx, split_feature=split_feature)
            content = (tree_num, node_num, left_child, right_child)
        else:
            content = (tree_num, node_num)

        send_message = Message(msg_type='continue_training',
                               sender=self.ID,
                               state=self.state,
                               receiver=[sender],
                               content=content)
        if sender == self.ID:
            self.callback_funcs_for_continue_training(send_message)
        else:
            self.comm_manager.send(send_message)

    def callback_funcs_for_continue_training(self, message: Message):
        if len(message.content) == 4:
            tree_num, node_num, left_child, right_child = message.content
            self.trainer.update_child(tree_num, node_num, left_child,
                                      right_child)
        else:
            tree_num, node_num = message.content

        self.train(tree_num=tree_num, node_num=node_num + 1)

    def callback_funcs_for_grad_and_hess(self, message: Message):
        tree_num, node_num, g, h, indicator = message.content
        improved_flag, split_info, best_gain = self.trainer._get_best_gain(
            tree_num, node_num, g, h, indicator)
        if 'feature_idx' in split_info and 'value_idx' in split_info:
            self.trainer.split_ref = split_info

        send_message = Message(msg_type='local_best_gain',
                               sender=self.ID,
                               state=self.state,
                               receiver=[message.sender],
                               content=(tree_num, node_num, best_gain,
                                        split_info, improved_flag))
        if message.sender == self.ID:
            self.callback_funcs_for_local_best_gain(send_message)
        else:
            self.comm_manager.send(send_message)

    def callback_funcs_for_local_best_gain(self, message: Message):
        tree_num, node_num, local_best_gain, split_info, improved_flag = \
            message.content
        client_id = message.sender
        self.msg_buffer['train'][client_id] = (local_best_gain, improved_flag,
                                               split_info)
        if len(self.msg_buffer['train']) == self.client_num:
            received_msg = copy.deepcopy(self.msg_buffer['train'])
            self.msg_buffer['train'].clear()
            max_gain, split_client_id, split_ref = \
                self.trainer.get_best_gain_from_msg(tree_num=tree_num,
                                                    node_num=node_num,
                                                    msg=received_msg)
            if max_gain is not None:
                self.model[tree_num][node_num].member = split_client_id
                split_child = True
                send_message = Message(msg_type='split',
                                       sender=self.ID,
                                       state=self.state,
                                       receiver=[split_client_id],
                                       content=(tree_num, node_num, split_ref,
                                                split_child))
                if split_client_id == self.ID:
                    self.callback_func_for_split(send_message)
                else:
                    self.comm_manager.send(send_message)
            else:
                self.trainer._set_weight_and_status(tree_num, node_num)

    # Bind method to instance
    client.train = types.MethodType(train, client)
    client.callback_func_for_split = types.MethodType(callback_func_for_split,
                                                      client)
    client.callback_funcs_for_continue_training = types.MethodType(
        callback_funcs_for_continue_training, client)
    client.callback_funcs_for_grad_and_hess = types.MethodType(
        callback_funcs_for_grad_and_hess, client)
    client.callback_funcs_for_local_best_gain = types.MethodType(
        callback_funcs_for_local_best_gain, client)
    client._find_and_send_split = types.MethodType(_find_and_send_split,
                                                   client)
    client._check_eval_finish = types.MethodType(_check_eval_finish, client)

    # Register handler functions
    client.register_handlers('split', client.callback_func_for_split)
    client.register_handlers('continue_training',
                             client.callback_funcs_for_continue_training)
    client.register_handlers('grad_and_hess',
                             client.callback_funcs_for_grad_and_hess)
    client.register_handlers('local_best_gain',
                             client.callback_funcs_for_local_best_gain)

    return client


def wrap_server_for_train(server):

    return server
