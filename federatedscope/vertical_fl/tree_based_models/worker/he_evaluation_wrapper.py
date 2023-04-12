import types
import logging
import numpy as np

from federatedscope.vertical_fl.loss.utils import get_vertical_loss
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_he_evaluation(client):
    """
    Use PHE to perform secure evaluation.
    For more details, please refer to the following papers:
        An Efficient and Robust System for Vertically Federated Random Forest
        (https://arxiv.org/pdf/2201.10761.pdf)
        Privacy Preserving Vertical Federated Learning for Tree-based Models
        (https://arxiv.org/pdf/2008.06170.pdf)
        Fed-EINI: An Efficient and Interpretable Inference Framework for
            Decision Tree Ensembles in Vertical Fed
        (https://arxiv.org/pdf/2105.09540.pdf)

    """
    def eval(self, tree_num):
        self.criterion = get_vertical_loss(loss_type=self._cfg.criterion.type,
                                           model_type=self._cfg.model.type)

        off_node_list = list()
        for node_num in range(2**self.model.max_depth - 1):
            if self.model[tree_num][node_num].status == 'off':
                off_node_list.append(node_num)
        if off_node_list:
            self.comm_manager.send(
                Message(
                    msg_type='off_node_list',
                    sender=self.ID,
                    state=self.state,
                    receiver=[
                        each
                        for each in list(self.comm_manager.neighbors.keys())
                        if each != self.server_id
                    ],
                    content=(tree_num, off_node_list)))

        if self.test_x is None:
            self.test_x, self.test_y = self._fetch_test_data()
            self.merged_test_result = list()
        self.test_result = np.zeros(self.test_x.shape[0])

        self.one_tree_weight_vector = self.iterate_for_weight_vector(
            tree_num, list(range(2**self.model.max_depth - 1)))
        if self._cfg.model.type in ['xgb_tree', 'gbdt_tree']:
            eta = self._cfg.train.optimizer.eta
        else:
            eta = 1.0
        self.one_tree_weight_vector = [
            eta * x for x in self.one_tree_weight_vector
        ]
        enc_one_tree_weight_vector = [
            self.public_key.encrypt(x) for x in self.one_tree_weight_vector
        ]
        indicator_array = self.get_test_result_for_one_tree(tree_num)
        enc_indicator_weight_array = \
            indicator_array * enc_one_tree_weight_vector
        self.comm_manager.send(
            Message(msg_type='enc_pred_result',
                    sender=self.ID,
                    state=self.state,
                    receiver=self.ID - 1,
                    content=(tree_num, enc_indicator_weight_array)))

    def callback_func_for_off_node_list(self, message: Message):
        tree_num, off_node_list = message.content
        for node_num in off_node_list:
            self.model[tree_num][node_num].status = 'off'

    def callback_func_for_enc_pred_result(self, message: Message):
        if self.test_x is None:
            self.test_x, self.test_y = self._fetch_test_data()
        tree_num, enc_indicator_weight_array = message.content
        indicator_array = self.get_test_result_for_one_tree(tree_num)
        enc_indicator_weight_array = \
            indicator_array * enc_indicator_weight_array
        if self.ID != 1:
            self.comm_manager.send(
                Message(msg_type='enc_pred_result',
                        sender=self.ID,
                        state=self.state,
                        receiver=self.ID - 1,
                        content=(tree_num, enc_indicator_weight_array)))
        else:
            enc_res = np.sum(enc_indicator_weight_array, axis=1)
            self.comm_manager.send(
                Message(msg_type='pred_result',
                        sender=self.ID,
                        state=self.state,
                        receiver=self.client_num,
                        content=(tree_num, enc_res)))

    def _fetch_test_data(self):
        test_x = self.data['test']['x']
        test_y = self.data['test']['y'] if 'y' in self.data['test'] else None

        return test_x, test_y

    def callback_func_for_pred_result(self, message: Message):
        tree_num, enc_res = message.content
        self.test_result = np.asarray(
            [self.private_key.decrypt(x) for x in enc_res])
        self.merged_test_result.append(self.test_result)
        if (
                tree_num + 1
        ) % self._cfg.eval.freq == 0 or \
                tree_num + 1 == self._cfg.model.num_of_trees:
            self._feedback_eval_metrics()
        self.eval_finish_flag = True
        self._check_eval_finish(tree_num)

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

    def callback_func_for_feature_importance(self, message: Message):
        state = message.state
        self.comm_manager.send(
            Message(msg_type='feature_importance',
                    sender=self.ID,
                    state=state,
                    receiver=[self.server_id],
                    content=self.feature_importance))

    def iterate_for_leaf_vector(self, x, tree_num, tree_node_list, flag):
        tree = self.model[tree_num]
        node_num = tree_node_list[0]
        feature_idx = tree[node_num].feature_idx
        if tree[node_num].status == 'off':
            return np.asarray([flag])
        else:
            if flag == 0:
                left_flag = right_flag = 0
            else:
                if feature_idx is None or tree[node_num].feature_value is None:
                    left_flag = right_flag = 1
                elif x[feature_idx] < tree[node_num].feature_value:
                    left_flag, right_flag = 1, 0
                else:
                    left_flag, right_flag = 0, 1
            subtree_size = int(np.log2(len(tree_node_list)))
            left_subtree_node_list = []
            right_subtree_node_list = []
            for i in range(1, subtree_size + 1):
                subtree_node_list = tree_node_list[2**i - 1:2**(i + 1) - 1]
                length = len(subtree_node_list)
                left_subtree_node_list.extend(subtree_node_list[:length // 2])
                right_subtree_node_list.extend(subtree_node_list[length // 2:])
            left_vector = self.iterate_for_leaf_vector(x, tree_num,
                                                       left_subtree_node_list,
                                                       left_flag)
            right_vector = self.iterate_for_leaf_vector(
                x, tree_num, right_subtree_node_list, right_flag)
            return np.concatenate((left_vector, right_vector))

    def iterate_for_weight_vector(self, tree_num, tree_node_list):
        tree = self.model[tree_num]
        node_num = tree_node_list[0]
        if tree[node_num].status == 'off':
            return np.asarray([tree[node_num].weight])
        else:
            subtree_size = int(np.log2(len(tree_node_list)))
            left_subtree_node_list = []
            right_subtree_node_list = []
            for i in range(1, subtree_size + 1):
                subtree_node_list = tree_node_list[2**i - 1:2**(i + 1) - 1]
                length = len(subtree_node_list)
                left_subtree_node_list.extend(subtree_node_list[:length // 2])
                right_subtree_node_list.extend(subtree_node_list[length // 2:])
            left_vector = self.iterate_for_weight_vector(
                tree_num, left_subtree_node_list)
            right_vector = self.iterate_for_weight_vector(
                tree_num, right_subtree_node_list)
            return np.concatenate((left_vector, right_vector))

    def get_test_result_for_one_tree(self, tree_num):
        res = [0] * self.test_x.shape[0]
        for i in range(len(self.test_x)):
            res[i] = self.iterate_for_leaf_vector(
                self.test_x[i],
                tree_num,
                list(range(2**self.model.max_depth - 1)),
                flag=1)
        return np.asarray(res)

    # Bind method to instance
    client.eval = types.MethodType(eval, client)
    client._fetch_test_data = types.MethodType(_fetch_test_data, client)
    client.iterate_for_leaf_vector = types.MethodType(iterate_for_leaf_vector,
                                                      client)
    client._feedback_eval_metrics = types.MethodType(_feedback_eval_metrics,
                                                     client)
    client.iterate_for_weight_vector = types.MethodType(
        iterate_for_weight_vector, client)
    client.get_test_result_for_one_tree = types.MethodType(
        get_test_result_for_one_tree, client)
    client.callback_func_for_off_node_list = types.MethodType(
        callback_func_for_off_node_list, client)
    client.callback_func_for_enc_pred_result = types.MethodType(
        callback_func_for_enc_pred_result, client)
    client.callback_func_for_pred_result = types.MethodType(
        callback_func_for_pred_result, client)
    client.callback_func_for_feature_importance = types.MethodType(
        callback_func_for_feature_importance, client)

    # Register handler functions
    client.register_handlers('off_node_list',
                             client.callback_func_for_off_node_list)
    client.register_handlers('enc_pred_result',
                             client.callback_func_for_enc_pred_result)
    client.register_handlers('pred_result',
                             client.callback_func_for_pred_result)
    client.register_handlers('ask_for_feature_importance',
                             client.callback_func_for_feature_importance)

    return client


def wrap_server_for_evaluation(server):
    def _check_and_save_result(self):

        state = max(self.msg_buffer['eval'].keys())
        buffer = self.msg_buffer['eval'][state]
        if len(buffer['feature_importance']
               ) == self.client_num and buffer['metrics'] is not None:
            self.state = state
            self.feature_importance = dict(
                sorted(buffer['feature_importance'].items(),
                       key=lambda x: x[0]))
            self.metrics = buffer['metrics']
            self._monitor.update_best_result(self.best_results,
                                             self.metrics,
                                             results_type='server_global_eval')
            self._monitor.add_items_to_best_result(
                self.best_results,
                self.feature_importance,
                results_type='feature_importance')
            formatted_logs = self._monitor.format_eval_res(
                self.metrics,
                rnd=self.state,
                role='Server #',
                forms=self._cfg.eval.report)
            formatted_logs['feature_importance'] = self.feature_importance
            logger.info(formatted_logs)

            if self.state + 1 == self._cfg.model.num_of_trees:
                self.terminate()

    def callback_func_for_feature_importance(self, message: Message):
        # Save the feature importance
        feature_importance = message.content
        sender = message.sender
        state = message.state
        if state not in self.msg_buffer['eval']:
            self.msg_buffer['eval'][state] = {}
            self.msg_buffer['eval'][state]['feature_importance'] = {}
            self.msg_buffer['eval'][state]['metrics'] = None
        self.msg_buffer['eval'][state]['feature_importance'].update(
            {str(sender): feature_importance})
        self._check_and_save_result()

    def callback_funcs_for_metrics(self, message: Message):
        state, metrics = message.state, message.content
        if state not in self.msg_buffer['eval']:
            self.msg_buffer['eval'][state] = {}
            self.msg_buffer['eval'][state]['feature_importance'] = {}
            self.msg_buffer['eval'][state]['metrics'] = None
        self.msg_buffer['eval'][state]['metrics'] = metrics
        self._check_and_save_result()

    # Bind method to instance
    server._check_and_save_result = types.MethodType(_check_and_save_result,
                                                     server)
    server.callback_func_for_feature_importance = types.MethodType(
        callback_func_for_feature_importance, server)
    server.callback_funcs_for_metrics = types.MethodType(
        callback_funcs_for_metrics, server)

    # Register handler functions
    server.register_handlers('feature_importance',
                             server.callback_func_for_feature_importance)
    server.register_handlers('eval_metric', server.callback_funcs_for_metrics)

    return server
