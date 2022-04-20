# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class ToyLRTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_standalone(self, cfg, make_global_eval=False):
        backup_cfg = cfg.clone()

        cfg.use_gpu = False
        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 20
        cfg.federate.make_global_eval = make_global_eval
        cfg.federate.client_num = 5
        cfg.eval.freq = 10
        cfg.data.type = 'toy'
        cfg.trainer.type = 'general'
        cfg.model.type = 'lr'

        return backup_cfg

    def test_toy_example_standalone(self):
        backup_cfg = self.set_config_standalone(global_cfg)
        setup_seed(global_cfg.seed)
        update_logger(global_cfg)

        data, modified_config = get_data(global_cfg.clone())
        global_cfg.merge_from_other_cfg(modified_config)

        self.assertIsNotNone(data)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(global_cfg),
                               client_class=get_client_cls(global_cfg),
                               config=global_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        global_cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(
            test_best_results["client_summarized_weighted_avg"]['test_loss'],
            0.3)

    def test_toy_example_standalone_global_eval(self):
        backup_cfg = self.set_config_standalone(global_cfg,
                                                make_global_eval=True)
        setup_seed(global_cfg.seed)
        update_logger(global_cfg)

        data, modified_config = get_data(global_cfg.clone())
        global_cfg.merge_from_other_cfg(modified_config)

        self.assertIsNotNone(data)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(global_cfg),
                               client_class=get_client_cls(global_cfg),
                               config=global_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        global_cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(test_best_results["server_global_eval"]['test_loss'],
                        0.3)


if __name__ == '__main__':
    unittest.main()
