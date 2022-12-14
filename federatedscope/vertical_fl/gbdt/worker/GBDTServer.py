from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.vertical_fl.gbdt.worker.Tree import Tree

import numpy as np
import logging

logger = logging.getLogger(__name__)


class GBDTServer(Server):
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
        super(GBDTServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        self.lambda_ = config.train.optimizer.lambda_
        self.gamma = config.train.optimizer.gamma
        self.num_of_trees = config.train.optimizer.num_of_trees
        self.max_tree_depth = config.train.optimizer.max_tree_depth

        self.num_of_parties = config.federate.client_num

        self.batch_size = config.data.batch_size

        self.feature_list = [0] + config.vertical_dims
        self.feature_partition = [
            self.feature_list[i + 1] - self.feature_list[i]
            for i in range(len(self.feature_list) - 1)
        ]
        self.total_num_of_feature = self.feature_list[-1]

        self.data = data

        self.tree_list = [
            Tree(self.max_tree_depth).tree for _ in range(self.num_of_trees)
        ]
        self.feature_importance_dict = dict()

        self.register_handlers('test_result',
                               self.callback_func_for_test_result)
        self.register_handlers('feature_importance',
                               self.callback_func_for_feature_importance)

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

    # TODO: merge the following two callback funcs
    def callback_func_for_feature_importance(self, message: Message):
        feature_importance = message.content
        self.feature_importance_dict[message.sender] = feature_importance
        if len(self.feature_importance_dict) == self.num_of_parties:
            self.feature_importance_dict = dict(
                sorted(self.feature_importance_dict.items(),
                       key=lambda x: x[0]))
            self._monitor.update_best_result(self.best_results,
                                             self.metrics,
                                             results_type='server_global_eval')
            self._monitor.add_items_to_best_result(
                self.best_results,
                self.feature_importance_dict,
                results_type='feature_importance')
            formatted_logs = self._monitor.format_eval_res(
                self.metrics,
                rnd=self.tree_num,
                role='Server #',
                forms=self._cfg.eval.report)
            formatted_logs['feature_importance'] = self.feature_importance_dict
            logger.info(formatted_logs)
            self.comm_manager.send(
                Message(msg_type='finish',
                        sender=self.ID,
                        receiver=list(
                            self.comm_manager.get_neighbors().keys()),
                        state=self.state,
                        content='None'))
            # jump out running
            self.state = self.total_round_num + 1

    def callback_func_for_test_result(self, message: Message):
        self.tree_num, self.metrics = message.content
