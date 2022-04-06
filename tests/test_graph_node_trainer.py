# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, setup_logger
from federatedscope.config import cfg, assert_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class NodeTrainerTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_node(self, cfg):
        backup_cfg = cfg.clone()

        cfg.use_gpu = True
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc', 'correct']

        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 50
        cfg.federate.client_num = 5

        cfg.data.root = 'test_data/'
        cfg.data.type = 'cora'
        cfg.data.batch_size = 1  # full batch train
        cfg.data.splitter = 'louvain'

        cfg.model.type = 'gcn'
        cfg.model.hidden = 64
        cfg.model.dropout = 0.5
        cfg.model.out_channels = 7

        cfg.optimizer.lr = 0.25
        cfg.optimizer.weight_decay = 0.0005
        cfg.optimizer.type = 'SGD'

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'nodefullbatch_trainer'

        return backup_cfg

    def test_node_standalone(self):
        backup_cfg = self.set_config_node(cfg)
        setup_seed(cfg.seed)
        setup_logger(cfg)

        data, modified_cfg = get_data(cfg.clone())
        cfg.merge_from_other_cfg(modified_cfg)
        assert_cfg(cfg)
        self.assertIsNotNone(data)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(cfg),
                               client_class=get_client_cls(cfg),
                               config=cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        cfg.merge_from_other_cfg(backup_cfg)
        self.assertGreater(
            test_best_results["client_summarized_weighted_avg"]['test_acc'],
            0.7)


if __name__ == '__main__':
    unittest.main()
