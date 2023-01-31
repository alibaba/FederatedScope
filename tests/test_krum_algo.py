# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class KrumAlgoTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_guassian_attack_no_defnece(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.device = 0
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc', 'loss_regular']

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 5
        cfg.federate.total_round_num = 20
        cfg.federate.sample_client_num = 10
        cfg.federate.client_num = 50

        cfg.data.root = 'test_data/'
        cfg.data.type = 'femnist'
        cfg.data.splits = [0.6, 0.2, 0.2]
        cfg.data.batch_size = 10
        cfg.data.subsample = 0.01
        cfg.data.transform = [['ToTensor'],
                              [
                                  'Normalize', {
                                      'mean': [0.1307],
                                      'std': [0.3081]
                                  }
                              ]]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 512
        cfg.model.out_channels = 62

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = 1,2,3,4,5

        return backup_cfg

    def test_GradAscent_femnist_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_guassian_attack_no_defnece(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertLess(test_best_results['server_global_eval']['test_acc'],
                           0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)


if __name__ == '__main__':
    unittest.main()
