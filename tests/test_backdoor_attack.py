# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class Backdoor_Attack(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_femnist(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.metrics = ['acc', 'correct', 'poison_attack_acc']

        cfg.early_stop.patience = 0
        cfg.federate.mode = 'standalone'
        cfg.train.batch_or_epoch = 'epoch'
        cfg.train.local_update_steps = 2
        cfg.federate.total_round_num = 10
        cfg.federate.sample_client_num = 20
        cfg.federate.client_num = 200

        cfg.data.root = 'test_data/'
        cfg.data.type = 'femnist'
        cfg.data.splits = [0.6, 0.2, 0.2]
        cfg.data.batch_size = 32
        cfg.data.subsample = 0.05
        cfg.data.transform = [['ToTensor']]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 62

        cfg.train.optimizer.lr = 0.1
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.attack.attack_method = 'backdoor'
        cfg.attack.attacker_id = -1
        cfg.attack.inject_round = 0
        cfg.attack.setting = 'fix'
        cfg.attack.freq = 10
        cfg.attack.label_type = 'dirty'
        cfg.attack.trigger_type = 'gridTrigger'
        cfg.attack.target_label_ind = 1
        cfg.attack.mean = [0.1307]
        cfg.attack.std = [0.3081]

        return backup_cfg

    def test_backdoor_edge_femnist_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_femnist(init_cfg)
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
        test_best_results = Fed_runner.run()
        print(test_best_results)

        # TODO: use a resonable metric
        self.assertGreater(
            test_best_results["client_summarized_weighted_avg"]['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)


if __name__ == '__main__':
    unittest.main()
