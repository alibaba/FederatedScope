# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class GradAscentTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_femnist(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.device = 0
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc', 'loss_regular']

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 5
        cfg.federate.total_round_num = 20
        cfg.federate.sample_client_num = 5
        cfg.federate.client_num = 10

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
        cfg.model.hidden = 2048
        cfg.model.out_channels = 62

        cfg.train.optimizer.lr = 0.001
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.attack.attack_method = 'GradAscent'
        cfg.attack.attacker_id = 5
        cfg.attack.inject_round = 0

        return backup_cfg

    def test_GradAscent_femnist_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_femnist(init_cfg)
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(init_cfg),
                               client_class=get_client_cls(init_cfg),
                               config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)

        # TODO: use a resonable metric
        self.assertLess(
            test_best_results["client_summarized_weighted_avg"]['test_loss'],
            600)
        # print(Fed_runner.client.keys())
        target_data_loss = Fed_runner.client[
            init_cfg.attack.attacker_id].trainer.ctx.target_data_loss
        self.assertIsNotNone(target_data_loss)
        self.assertIn(init_cfg.attack.attacker_id, Fed_runner.client.keys())

        init_cfg.merge_from_other_cfg(backup_cfg)


if __name__ == '__main__':
    unittest.main()
