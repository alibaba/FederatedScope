# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, setup_logger
from federatedscope.config import cfg
from federatedscope.core.DAIL_fed_api import DAILFed
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class MFTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_movielens1m(self, cfg):
        backup_cfg = cfg.clone()

        cfg.use_gpu = True
        cfg.device = 0
        cfg.early_stopping = 100
        cfg.eval.freq = 5
        cfg.eval.metrics = []

        cfg.federate.mode = 'standalone'
        cfg.federate.local_update_steps = 10
        cfg.federate.total_round_num = 5
        cfg.federate.client_num = 5

        cfg.data.root = 'test_data/'
        cfg.data.type = 'movielens1m'
        cfg.data.batch_size = 8

        cfg.model.type = 'MFNet'
        cfg.model.hidden = 20

        cfg.optimizer.lr = 0.1
        cfg.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'MSELoss'
        cfg.trainer.type = 'mftrainer'
        cfg.seed = 123

        cfg.sgdmf.use = True

        return backup_cfg

    def test_mf_standalone(self):
        backup_cfg = self.set_config_movielens1m(cfg)
        setup_seed(cfg.seed)
        setup_logger(cfg)

        data, modified_cfg = get_data(cfg.clone())
        cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = DAILFed(data=data,
                             server_class=get_server_cls(cfg),
                             client_class=get_client_cls(cfg),
                             config=cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_results = Fed_runner.run()
        cfg.merge_from_other_cfg(backup_cfg)
        print(test_results)
        self.assertLess(test_results['client_individual']['avg_loss'], 20)


if __name__ == '__main__':
    unittest.main()
