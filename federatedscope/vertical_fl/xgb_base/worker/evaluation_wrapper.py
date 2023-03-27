import types
import logging
import numpy as np

from federatedscope.vertical_fl.loss.utils import get_vertical_loss
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_evaluation(client):
    def eval(self, tree_num):
        self.criterion = get_vertical_loss(loss_type=self._cfg.criterion.type,
                                           model_type=self._cfg.model.type)
        if self.test_x is None:
            self.test_x, self.test_y = self._fetch_test_data()
            self.merged_test_result = list()
        self.test_result = np.zeros(self.test_x.shape[0])
        self.model[tree_num][0].indicator = np.ones(self.test_x.shape[0])
        self._test_for_node(tree_num, node_num=0)

    def _fetch_test_data(self):
        test_x = self.data['test']['x']
        test_y = self.data['test']['y'] if 'y' in self.data['test'] else None

        return test_x, test_y

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
            self.merged_test_result.append(self.test_result)
            if (
                    tree_num + 1
            ) % self._cfg.eval.freq == 0 or \
                    tree_num + 1 == self._cfg.model.num_of_trees:
                self._feedback_eval_metrics()
            self.eval_finish_flag = True
            self._check_eval_finish(tree_num)
        # The client owns the weight
        elif self.model[tree_num][node_num].weight:
            if self._cfg.model.type in ['xgb_tree', 'gbdt_tree']:
                eta = self._cfg.train.optimizer.eta
            else:
                eta = 1.0
            self.test_result += self.model[tree_num][
                node_num].indicator * self.model[tree_num][
                    node_num].weight * eta
            self._test_for_node(tree_num, node_num + 1)
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

    def callback_func_for_split_request(self, message: Message):
        if self.test_x is None:
            self.test_x, self.test_y = self._fetch_test_data()
        tree_num, node_num = message.content
        sender = message.sender
        feature_idx = self.model[tree_num][node_num].feature_idx
        feature_value = self.model[tree_num][node_num].feature_value
        left_child, right_child = self.model[tree_num].split_childern(
            self.test_x[:, feature_idx], feature_value)
        send_message = Message(msg_type='split_result',
                               sender=self.ID,
                               state=self.state,
                               receiver=[sender],
                               content=(tree_num, node_num, left_child,
                                        right_child))
        if sender == self.ID:
            self.callback_func_for_split_result(send_message)
        else:
            self.comm_manager.send(send_message)

    def callback_func_for_split_result(self, message: Message):
        tree_num, node_num, left_child, right_child = message.content
        self.model[tree_num][2 * node_num + 1].indicator = self.model[
            tree_num][node_num].indicator * left_child
        self.model[tree_num][2 * node_num + 2].indicator = self.model[
            tree_num][node_num].indicator * right_child
        self._test_for_node(tree_num, node_num + 1)

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
