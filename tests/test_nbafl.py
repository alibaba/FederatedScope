# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, setup_logger
from federatedscope.config import cfg, assert_cfg
from federatedscope.core.DAIL_fed_api import DAILFed
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class NbAFLTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_femnist(self, cfg):
        backup_cfg = cfg.clone()

        cfg.use_gpu = True
        cfg.device = 0
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc', 'loss_regular']

        cfg.federate.mode = 'standalone'
        cfg.federate.local_update_steps = 5
        cfg.federate.total_round_num = 20
        cfg.federate.sample_client_num = 5

        cfg.data.root = 'test_data/'
        cfg.data.type = 'femnist'
        cfg.data.splits = [0.6, 0.2, 0.2]
        cfg.data.batch_size = 10
        cfg.data.subsample = 0.01

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 62

        cfg.optimizer.lr = 0.001
        cfg.optimizer.weight_decay = 0.0
        cfg.optimizer.grad_clip = 5.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.nbafl.use = True
        cfg.nbafl.w_clip = 0.1
        cfg.nbafl.mu = 0.1
        cfg.nbafl.constant = 2.
        cfg.nbafl.epsilon = 50.

        return backup_cfg

    def test_nbafl_standalone(self):
        backup_cfg = self.set_config_femnist(cfg)
        setup_seed(cfg.seed)
        setup_logger(cfg)

        data, modified_cfg = get_data(cfg.clone())
        cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)
        # Run on first 10 clients
        cfg.merge_from_list(['federate.client_num', 10])
        assert_cfg(cfg)

        Fed_runner = DAILFed(data=data,
                             server_class=get_server_cls(cfg),
                             client_class=get_client_cls(cfg),
                             config=cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(test_best_results["client_summarized_weighted_avg"]['test_loss'], 500)


if __name__ == '__main__':
    unittest.main()
