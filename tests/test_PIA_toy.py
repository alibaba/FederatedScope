# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class PIA_ToyLRTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_standalone(self, cfg):
        backup_cfg = cfg.clone()

        cfg.use_gpu = False
        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 20
        cfg.federate.client_num = 5
        cfg.eval.freq = 10
        cfg.data.type = 'toy'
        cfg.trainer.type = 'general'
        cfg.model.type = 'lr'

        cfg.attack.attack_method = 'PassivePIA'
        cfg.attack.classifier_PIA = 'svm'

        return backup_cfg

    def test_PIA_toy_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_standalone(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_config = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_config)

        self.assertIsNotNone(data)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(init_cfg),
                               client_class=get_client_cls(init_cfg),
                               config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)

        self.assertLess(
            test_best_results["client_summarized_weighted_avg"]['test_loss'],
            0.3)
        self.assertIsNotNone(Fed_runner.server.pia_results)

        init_cfg.merge_from_other_cfg(backup_cfg)


if __name__ == '__main__':
    unittest.main()
