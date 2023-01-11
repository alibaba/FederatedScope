import types
import logging

from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_train(client):
    def train(self, tree_num, node_num=None):
        if node_num is None:
            logger.info(f'----------- Building a new tree (Tree '
                        f'#{tree_num}) -------------')
            self.state = tree_num
            finish_flag, results = self.trainer.train(
                feature_order=self.merged_feature_order, tree_num=tree_num)
        else:
            assert node_num is not None
            finish_flag, results = self.trainer.train(tree_num=tree_num,
                                                      node_num=node_num)

        if not finish_flag:
            split_ref, tree_num, node_num = results
            self._find_and_send_split(split_ref, tree_num, node_num)
        else:
            # Report evaluation results
            self.eval(tree_num)

            if tree_num + 1 < self._cfg.model.num_of_trees:
                self.train(tree_num + 1)

    def _find_and_send_split(self, split_ref, tree_num, node_num):
        for index, dim in enumerate(self._cfg.vertical.dims):
            if split_ref['feature_idx'] < dim:
                prefix = self._cfg.vertical.dims[index -
                                                 1] if index != 0 else 0
                client_id = index + 1
                self.model[tree_num][node_num].member = client_id
                self.model[tree_num][
                    node_num].feature_idx = split_ref['feature_idx'] - prefix

                self.comm_manager.send(
                    Message(msg_type='split',
                            sender=self.ID,
                            state=self.state,
                            receiver=[client_id],
                            content=(tree_num, node_num,
                                     split_ref['feature_idx'] - prefix,
                                     split_ref['value_idx'])))
                break

    def callback_func_for_split(self, message: Message):
        tree_num, node_num, feature_idx, value_idx = message.content
        sender = message.sender
        self.state = message.state
        self.feature_importance[feature_idx] += 1

        instance_idx = self.feature_order[feature_idx][value_idx]
        feature_value = self.trainer.batch_x[instance_idx, feature_idx]

        self.model[tree_num][node_num].feature_idx = feature_idx
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

    # Register handler functions
    client.register_handlers('split', client.callback_func_for_split)
    client.register_handlers('continue_training',
                             client.callback_funcs_for_continue_training)

    return client


def wrap_server_for_train(server):

    return server
