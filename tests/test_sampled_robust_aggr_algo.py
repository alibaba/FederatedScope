# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class sampled_aggr_AlgoTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_oracle_fedavg(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop.patience = 5
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.count_flops = False
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_loss'

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 1
        cfg.train.batch_or_epoch = 'epoch'
        cfg.federate.total_round_num = 200
        cfg.federate.sample_client_num = 20
        cfg.federate.client_num = 50
        cfg.federate.make_global_eval = True
        cfg.federate.merge_test_data = True

        cfg.data.root = 'data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args=[{'download': True}]
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 64
        cfg.data.subsample = 0.01
        cfg.data.transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.test_transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.splitter='lda'
        cfg.data.splitter_args= [{'alpha': 0.1}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 10
        cfg.model.dropout = 0

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 12345

        cfg.aggregator.fedavg.use = True

        return backup_cfg

    
    def set_config_multikrum(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop.patience = 5
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.count_flops = False
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_loss'

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 1
        cfg.train.batch_or_epoch = 'epoch'
        cfg.federate.total_round_num = 200
        cfg.federate.sample_client_num = 40
        cfg.federate.client_num = 200
        cfg.federate.make_global_eval = True
        cfg.federate.merge_test_data = True

        cfg.data.root = 'data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args=[{'download': True}]
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 64
        cfg.data.subsample = 0.01
        cfg.data.transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.test_transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.splitter='lda'
        cfg.data.splitter_args= [{'alpha': 0.1}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 10
        cfg.model.dropout = 0

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 12345

        cfg.aggregator.byzantine_node_num = 40
        cfg.aggregator.krum.use = True
        cfg.aggregator.krum.agg_num = 32

        return backup_cfg

    def set_config_median(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop.patience = 5
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.count_flops = False
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_loss'

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 1
        cfg.train.batch_or_epoch = 'epoch'
        cfg.federate.total_round_num = 200
        cfg.federate.sample_client_num = 40
        cfg.federate.client_num = 200
        cfg.federate.make_global_eval = True
        cfg.federate.merge_test_data = True

        cfg.data.root = 'data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args=[{'download': True}]
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 64
        cfg.data.subsample = 0.01
        cfg.data.transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.test_transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.splitter='lda'
        cfg.data.splitter_args= [{'alpha': 0.1}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 10
        cfg.model.dropout = 0

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 12345

        cfg.aggregator.client_sampled_ratio=cfg.federate.client_num/cfg.federate.total_round_num
        cfg.aggregator.byzantine_node_num = 40
        cfg.aggregator.median.use = True

        return backup_cfg
    
    def set_config_trimmedmean(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop.patience = 5
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.count_flops = False
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_loss'

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 1
        cfg.train.batch_or_epoch = 'epoch'
        cfg.federate.total_round_num = 200
        cfg.federate.sample_client_num = 40
        cfg.federate.client_num = 200
        cfg.federate.make_global_eval = True
        cfg.federate.merge_test_data = True

        cfg.data.root = 'data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args=[{'download': True}]
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 64
        cfg.data.subsample = 0.01
        cfg.data.transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.test_transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.splitter='lda'
        cfg.data.splitter_args= [{'alpha': 0.1}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 10
        cfg.model.dropout = 0

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 12345

        cfg.aggregator.byzantine_node_num = 40
        cfg.aggregator.trimmedmean.use = True
        cfg.aggregator.trimmedmean.excluded_ratio = 0.1

        return backup_cfg

    def set_config_bulyan(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop.patience = 5
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.count_flops = False
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_loss'

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 1
        cfg.train.batch_or_epoch = 'epoch'
        cfg.federate.total_round_num = 200
        cfg.federate.sample_client_num = 40
        cfg.federate.client_num = 200
        cfg.federate.make_global_eval = True
        cfg.federate.merge_test_data = True

        cfg.data.root = 'data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args=[{'download': True}]
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 64
        cfg.data.subsample = 0.01
        cfg.data.transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.test_transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.splitter='lda'
        cfg.data.splitter_args= [{'alpha': 0.1}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 10
        cfg.model.dropout = 0

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 12345

        cfg.aggregator.bulyan.use = True
        cfg.aggregator.byzantine_node_num = 40
        return backup_cfg


    def set_config_sample(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop.patience = 5
        cfg.device = 0
        cfg.eval.freq = 1
        cfg.eval.count_flops = False
        cfg.eval.metrics = ['acc', 'correct']
        cfg.eval.best_res_update_round_wise_key = 'test_loss'

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 1
        cfg.train.batch_or_epoch = 'epoch'
        cfg.federate.total_round_num = 200
        cfg.federate.sample_client_num = 40
        cfg.federate.client_num = 200
        cfg.federate.make_global_eval = True
        cfg.federate.merge_test_data = True

        cfg.data.root = 'data/'
        cfg.data.type = 'CIFAR10@torchvision'
        cfg.data.args=[{'download': True}]
        cfg.data.splits = [0.8, 0.1, 0.1]
        cfg.data.batch_size = 64
        cfg.data.subsample = 0.01
        cfg.data.transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.test_transform=[['ToTensor'], ['Normalize', {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}]]
        cfg.data.splitter='lda'
        cfg.data.splitter_args= [{'alpha': 0.1}]

        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 10
        cfg.model.dropout = 0

        cfg.train.optimizer.lr = 0.01
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        cfg.seed = 12345

        cfg.aggregator.byzantine_node_num = 40
        cfg.aggregator.client_sampled_ratio=cfg.federate.client_num/cfg.federate.total_round_num
        cfg.aggregator.sampled_robust_aggregator.use = True
        cfg.aggregator.sampled_robust_aggregator.krum_agg_num=32
        cfg.aggregator.sampled_robust_aggregator.trimmedmean_excluded_ratio=0.1
        cfg.aggregator.sampled_robust_aggregator.candidate=['krum','median','trimmedmean','bulyan']
        return backup_cfg

    def test_0_oracal_fedavg(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_oracle_fedavg(init_cfg)
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
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)


    def test_1_multikrum(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_multikrum(init_cfg)
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
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)

    def test_3_median(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_median(init_cfg)
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
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)

    def test_4_trimmedmean(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_trimmedmean(init_cfg)
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
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)

    def test_5_bulyan(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_bulyan(init_cfg)
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
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)

    def test_6_sample(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_sample(init_cfg)
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
        self.assertLess(
            test_best_results['client_summarized_weighted_avg']['test_acc'],
            0.1)
        init_cfg.merge_from_other_cfg(backup_cfg)
        

if __name__ == '__main__':
    unittest.main()
