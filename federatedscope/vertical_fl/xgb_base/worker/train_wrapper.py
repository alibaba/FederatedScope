import types
import logging

from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_train(client):
    def train(self, tree_num, node_num=None, feature_order_info=None):
        if node_num is None:
            logger.info(f'----------- Building a new tree (Tree '
                        f'#{tree_num}) -------------')
            finish_flag, results = self.trainer.train(
                feature_order_info=feature_order_info, tree_num=tree_num)
        else:
            assert node_num is not None
            finish_flag, results = self.trainer.train(tree_num=tree_num,
                                                      node_num=node_num)

        if not finish_flag:
            split_ref, tree_num, node_num = results
            self._find_and_send_split(split_ref, tree_num, node_num)
        else:
            # Report evaluation results
            self.eval_finish_flag = False
            self.eval(tree_num)
            self._check_eval_finish(tree_num)

    def _check_eval_finish(self, tree_num):
        if self.eval_finish_flag:
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

                self.comm_manager.send(
                    Message(msg_type='split',
                            sender=self.ID,
                            state=self.state,
                            receiver=[client_id],
                            content=(tree_num, node_num,
                                     split_ref['feature_idx'] - accum_dim,
                                     split_ref['value_idx'])))
                break
            else:
                accum_dim += dim

    def callback_func_for_split(self, message: Message):
        tree_num, node_num, feature_idx, value_idx = message.content
        sender = message.sender
        self.state = message.state
        abs_feature_idx = self.trainer.get_abs_feature_idx(feature_idx)
        self.feature_importance[abs_feature_idx] += 1

        feature_value = self.trainer.get_feature_value(feature_idx, value_idx)

        self.model[tree_num][node_num].feature_idx = abs_feature_idx
        self.model[tree_num][node_num].feature_value = feature_value

        self.comm_manager.send(
            Message(msg_type='continue_training',
                    sender=self.ID,
                    state=self.state,
                    receiver=[sender],
                    content=(tree_num, node_num)))

    def callback_funcs_for_continue_training(self, message: Message):
        tree_num, node_num = message.content
        self.train(tree_num=tree_num, node_num=node_num + 1)

    # Bind method to instance
    client.train = types.MethodType(train, client)
    client.callback_func_for_split = types.MethodType(callback_func_for_split,
                                                      client)
    client.callback_funcs_for_continue_training = types.MethodType(
        callback_funcs_for_continue_training, client)
    client._find_and_send_split = types.MethodType(_find_and_send_split,
                                                   client)
    client._check_eval_finish = types.MethodType(_check_eval_finish, client)

    # Register handler functions
    client.register_handlers('split', client.callback_func_for_split)
    client.register_handlers('continue_training',
                             client.callback_funcs_for_continue_training)

    return client


def wrap_server_for_train(server):

    return server
