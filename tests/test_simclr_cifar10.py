# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls

SAMPLE_CLIENT_NUM = 5


class SimCLR_CIFAR10Test(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_simclr_cifar10(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 5
        cfg.eval.metrics = ['loss']
        cfg.eval.split = ['val', 'test']

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 5
        cfg.train.batch_or_epoch = 'batch'
        cfg.federate.total_round_num = 20
        cfg.federate.sample_client_num = 5

        cfg.data.root = 'test_data/'
        cfg.data.type = 'Cifar4CL'
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 256
        cfg.data.splitter = 'lda'
        cfg.data.splitter_args = [{'alpha': 0.1}]
        cfg.data.num_workers = 4
        cfg.data.subsample = 1.0

        cfg.model.type = 'SimCLR'
        cfg.model.hidden = 256
        cfg.model.out_channels = 1

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0001
        cfg.train.optimizer.momentum = 0.9

        cfg.criterion.type = 'NT_xentloss'
        cfg.trainer.type = 'cltrainer'
        cfg.seed = 1

        return backup_cfg

    def test_simclr_cifar10_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_simclr_cifar10(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)
        self.assertEqual(init_cfg.federate.sample_client_num,
                         SAMPLE_CLIENT_NUM)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(init_cfg),
                               client_class=get_client_cls(init_cfg),
                               config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(
            test_best_results["client_summarized_weighted_avg"]['test_loss'],
            100)


if __name__ == '__main__':
    unittest.main()
