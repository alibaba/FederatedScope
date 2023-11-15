# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, \
    get_client_cls


class TrainerCfgTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_trainer_cfg_test(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc', 'loss_regular']

        cfg.federate.mode = 'standalone'
        cfg.data.root = 'test_data/'
        cfg.data.type = 'femnist'
        cfg.data.splits = [0.6, 0.2, 0.2]
        cfg.dataloader.batch_size = 10
        cfg.data.subsample = 0.05
        cfg.data.transform = [['ToTensor'],
                              [
                                  'Normalize', {
                                      'mean': [0.9637],
                                      'std': [0.1592]
                                  }
                              ]]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 62

        cfg.train.optimizer.lr = 0.001
        cfg.train.optimizer.weight_decay = 0.0
        cfg.train.batch_or_epoch = 'epoch'
        cfg.grad.grad_clip = 5.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        return backup_cfg

    def test_trainer_cfg(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_trainer_cfg_test(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)

        num_train_batch = Fed_runner.client[1].trainer.ctx.num_train_batch
        new_cfg = init_cfg.clone()
        new_cfg.dataloader.batch_size = 64
        Fed_runner.client[1].trainer.cfg = new_cfg
        new_num_train_batch = Fed_runner.client[1].trainer.ctx.num_train_batch
        self.assertLess(new_num_train_batch, num_train_batch)

        init_cfg.merge_from_other_cfg(backup_cfg)


if __name__ == '__main__':
    unittest.main()
