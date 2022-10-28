# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class AsynCIFAR10Test(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_cifar10_goalAchieved_afterReceiving(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 5
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_acc'

        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 40
        cfg.federate.sample_client_num = 13
        cfg.federate.merge_test_data = True
        cfg.federate.share_local_model = False
        cfg.federate.client_num = 200
        cfg.federate.sampler = 'group'
        cfg.federate.resource_info_file = 'test_data/client_device_capacity'

        cfg.data.root = 'test_data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args = [{'download': True}]
        cfg.data.splits = [0.8, 0.2, 0.2]
        cfg.data.batch_size = 10
        cfg.data.subsample = 0.2
        cfg.data.num_workers = 0
        cfg.data.transform = [['ToTensor'],
                              [
                                  'Normalize', {
                                      'mean': [0.4914, 0.4822, 0.4465],
                                      'std': [0.247, 0.243, 0.261]
                                  }
                              ]]
        cfg.data.splitter = 'lda'
        cfg.data.splitter_args = [{'alpha': 0.2}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 128
        cfg.model.out_channels = 10

        cfg.train.local_update_steps = 2
        cfg.train.batch_or_epoch = 'batch'
        cfg.train.optimizer.lr = 0.1
        cfg.train.optimizer.weight_decay = 0.0
        cfg.grad.grad_clip = 5.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.asyn.use = True
        cfg.asyn.overselection = False
        cfg.asyn.staleness_discount_factor = 0.2
        cfg.asyn.aggregator = 'goal_achieved'
        cfg.asyn.broadcast_manner = 'after_receiving'
        cfg.asyn.min_received_num = 10
        cfg.asyn.staleness_toleration = 5

        return backup_cfg

    def set_config_cifar10_timeUp_afterAggregating(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 5
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_acc'

        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 40
        cfg.federate.sample_client_num = 13
        cfg.federate.merge_test_data = True
        cfg.federate.share_local_model = False
        cfg.federate.client_num = 200
        cfg.federate.sampler = 'uniform'
        cfg.federate.resource_info_file = 'test_data/client_device_capacity'

        cfg.data.root = 'test_data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args = [{'download': True}]
        cfg.data.splits = [0.8, 0.2, 0.2]
        cfg.data.batch_size = 10
        cfg.data.subsample = 0.2
        cfg.data.num_workers = 0
        cfg.data.transform = [['ToTensor'],
                              [
                                  'Normalize', {
                                      'mean': [0.4914, 0.4822, 0.4465],
                                      'std': [0.247, 0.243, 0.261]
                                  }
                              ]]
        cfg.data.splitter = 'lda'
        cfg.data.splitter_args = [{'alpha': 0.2}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 128
        cfg.model.out_channels = 10

        cfg.train.local_update_steps = 2
        cfg.train.batch_or_epoch = 'batch'
        cfg.train.optimizer.lr = 0.1
        cfg.train.optimizer.weight_decay = 0.0
        cfg.grad.grad_clip = 5.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.asyn.use = True
        cfg.asyn.overselection = False
        cfg.asyn.staleness_discount_factor = 0.2
        cfg.asyn.aggregator = 'time_up'
        cfg.asyn.time_budget = 10
        cfg.asyn.broadcast_manner = 'after_aggregating'
        cfg.asyn.min_received_num = 10
        cfg.asyn.staleness_toleration = 5

        return backup_cfg

    def set_config_cifar10_overselection(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 5
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_acc'

        cfg.federate.mode = 'standalone'
        cfg.federate.total_round_num = 40
        cfg.federate.sample_client_num = 13
        cfg.federate.merge_test_data = True
        cfg.federate.share_local_model = False
        cfg.federate.client_num = 200
        cfg.federate.sampler = 'uniform'
        cfg.federate.resource_info_file = 'test_data/client_device_capacity'

        cfg.data.root = 'test_data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args = [{'download': True}]
        cfg.data.splits = [0.8, 0.2, 0.2]
        cfg.data.batch_size = 10
        cfg.data.subsample = 0.2
        cfg.data.num_workers = 0
        cfg.data.transform = [['ToTensor'],
                              [
                                  'Normalize', {
                                      'mean': [0.4914, 0.4822, 0.4465],
                                      'std': [0.247, 0.243, 0.261]
                                  }
                              ]]
        cfg.data.splitter = 'lda'
        cfg.data.splitter_args = [{'alpha': 0.2}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 128
        cfg.model.out_channels = 10

        cfg.train.local_update_steps = 2
        cfg.train.batch_or_epoch = 'batch'
        cfg.train.optimizer.lr = 0.1
        cfg.train.optimizer.weight_decay = 0.0
        cfg.grad.grad_clip = 5.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 123

        cfg.asyn.use = True
        cfg.asyn.overselection = True
        cfg.asyn.staleness_discount_factor = 0.2
        cfg.asyn.aggregator = 'goal_achieved'
        cfg.asyn.broadcast_manner = 'after_aggregating'
        cfg.asyn.min_received_num = 10
        cfg.asyn.staleness_toleration = 0

        return backup_cfg

    def test_asyn_cifar10_goalAchieved_afterReceiving(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_cifar10_goalAchieved_afterReceiving(
            init_cfg)
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
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertGreater(test_best_results['server_global_eval']['test_acc'],
                           0.15)

    def test_asyn_cifar10_timeUp_afterAggregating(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_cifar10_timeUp_afterAggregating(init_cfg)
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
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertGreater(test_best_results['server_global_eval']['test_acc'],
                           0.15)

    def test_asyn_cifar10_overselection(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_cifar10_overselection(init_cfg)
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
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertGreater(test_best_results['server_global_eval']['test_acc'],
                           0.15)


if __name__ == '__main__':
    unittest.main()
