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
        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 20
        cfg.federate.make_global_eval = make_global_eval
        cfg.federate.client_num = 5
        cfg.federate.unseen_clients_rate = 0.2  # 20% unseen clients
        cfg.eval.freq = 10
        cfg.data.type = 'toy'
        cfg.trainer.type = 'general'
        cfg.model.type = 'lr'

    def test_toy_example_standalone(self):
        init_cfg = global_cfg.clone()
        self.set_config_standalone(init_cfg)

        setup_seed(init_cfg.seed)
        update_logger(init_cfg)

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
        self.assertLess(
            test_best_results["unseen_client_summarized_weighted_avg"]
            ['test_loss'], 0.3)


if __name__ == '__main__':
    unittest.main()
