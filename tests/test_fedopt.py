# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, setup_logger
from federatedscope.config import cfg, assert_cfg
from federatedscope.core.DAIL_fed_api import DAILFed
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class FedOptTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_fedopt(self, cfg):
        backup_cfg = cfg.clone()

        cfg.use_gpu = True
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc']

        cfg.federate.mode = 'standalone'
        cfg.federate.local_update_steps = 5
        cfg.federate.total_round_num = 20
        cfg.federate.sample_client_num = 5
        cfg.federate.client_num = 10
        cfg.federate.method = 'FedOpt'

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

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.fedopt.lr_server = 1.

        return backup_cfg

    def test_fedopt_standalone(self):
        backup_cfg = self.set_config_fedopt(cfg)
        setup_seed(cfg.seed)
        setup_logger(cfg)

        data, modified_cfg = get_data(cfg.clone())
        cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        assert_cfg(cfg)

        Fed_runner = DAILFed(data=data,
                             server_class=get_server_cls(cfg),
                             client_class=get_client_cls(cfg),
                             config=cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_results = Fed_runner.run()
        cfg.merge_from_other_cfg(backup_cfg)

        self.assertLess(
            test_results['client_summarized_weighted_avg']['test_loss'], 600)


if __name__ == '__main__':
    unittest.main()
