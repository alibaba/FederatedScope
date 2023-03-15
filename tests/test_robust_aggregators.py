# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class robust_aggr_AlgoTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    
    def set_config_multikrum(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.merge_from_file('federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml')

        cfg.federate.client_num = 50
        cfg.federate.total_round_num = 50 
        cfg.aggregator.byzantine_node_num = 10
        cfg.aggregator.krum.use = True
        cfg.aggregator.krum.agg_num = 30
        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = [i+1 for i in range(cfg.aggregator.byzantine_node_num)]

        return backup_cfg


    def set_config_median(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.merge_from_file('federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml')
        cfg.federate.client_num = 50
        cfg.federate.total_round_num = 50

        cfg.aggregator.byzantine_node_num = 10
        cfg.aggregator.median.use = True

        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = [i+1 for i in range(cfg.aggregator.byzantine_node_num)]

        return backup_cfg

    def set_config_trimmedmean(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.merge_from_file('federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml')
        cfg.federate.client_num = 50
        cfg.federate.total_round_num = 50

        cfg.aggregator.byzantine_node_num = 10
        cfg.aggregator.trimmedmean.use = True
        cfg.aggregator.trimmedmean.excluded_ratio = 0.2

        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = [i+1 for i in range(cfg.aggregator.byzantine_node_num)]

        return backup_cfg
    
    def set_config_bulyan(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.merge_from_file('federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml')

        cfg.federate.client_num = 50
        cfg.federate.total_round_num = 50

        cfg.aggregator.bulyan.use = True
        cfg.aggregator.byzantine_node_num = 10

        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = [i+1 for i in range(cfg.aggregator.byzantine_node_num)]
        return backup_cfg
    
    def set_config_normbounding(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.merge_from_file('federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml')

        cfg.federate.client_num = 50
        cfg.federate.total_round_num = 50

        cfg.aggregator.normbounding.use = True
        cfg.aggregator.normbounding.norm_bound = 5
        cfg.aggregator.byzantine_node_num = 10

        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = [i+1 for i in range(cfg.aggregator.byzantine_node_num)]
        return backup_cfg

    def set_config_fltrust(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.merge_from_file('federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml')

        cfg.federate.client_num = 50
        cfg.federate.total_round_num = 50

        cfg.aggregator.fltrust.use = True
        cfg.aggregator.byzantine_node_num = 10
        cfg.data.root_dataset_need = True
        cfg.eval.best_res_update_round_wise_key = 'val_loss'

        cfg.attack.attack_method = 'gaussian_noise'
        cfg.attack.attacker_id = [i+1 for i in range(cfg.aggregator.byzantine_node_num)]
        return backup_cfg




    # def test_0_multikrum(self):
    #     init_cfg = global_cfg.clone()
    #     backup_cfg = self.set_config_multikrum(init_cfg)
    #     setup_seed(init_cfg.seed)
    #     update_logger(init_cfg, True)

    #     data, modified_cfg = get_data(init_cfg.clone())
    #     init_cfg.merge_from_other_cfg(modified_cfg)
    #     self.assertIsNotNone(data)

    #     Fed_runner = get_runner(data=data,
    #                             server_class=get_server_cls(init_cfg),
    #                             client_class=get_client_cls(init_cfg),
    #                             config=init_cfg.clone())
    #     self.assertIsNotNone(Fed_runner)
    #     test_best_results = Fed_runner.run()
    #     print(test_best_results)
    #     init_cfg.merge_from_other_cfg(backup_cfg)
    #     self.assertLess(
    #         test_best_results['client_summarized_weighted_avg']['test_acc'],
    #         0.7)
    #     init_cfg.merge_from_other_cfg(backup_cfg)

    # def test_1_median(self):
    #     init_cfg = global_cfg.clone()
    #     backup_cfg = self.set_config_median(init_cfg)
    #     setup_seed(init_cfg.seed)
    #     update_logger(init_cfg, True)

    #     data, modified_cfg = get_data(init_cfg.clone())
    #     init_cfg.merge_from_other_cfg(modified_cfg)
    #     self.assertIsNotNone(data)

    #     Fed_runner = get_runner(data=data,
    #                             server_class=get_server_cls(init_cfg),
    #                             client_class=get_client_cls(init_cfg),
    #                             config=init_cfg.clone())
    #     self.assertIsNotNone(Fed_runner)
    #     test_best_results = Fed_runner.run()
    #     print(test_best_results)
    #     init_cfg.merge_from_other_cfg(backup_cfg)
    #     self.assertLess(
    #         test_best_results['client_summarized_weighted_avg']['test_acc'],
    #         0.7)
    #     init_cfg.merge_from_other_cfg(backup_cfg)

    # def test_2_trimmedmean(self):
    #     init_cfg = global_cfg.clone()
    #     backup_cfg = self.set_config_trimmedmean(init_cfg)
    #     setup_seed(init_cfg.seed)
    #     update_logger(init_cfg, True)

    #     data, modified_cfg = get_data(init_cfg.clone())
    #     init_cfg.merge_from_other_cfg(modified_cfg)
    #     self.assertIsNotNone(data)

    #     Fed_runner = get_runner(data=data,
    #                             server_class=get_server_cls(init_cfg),
    #                             client_class=get_client_cls(init_cfg),
    #                             config=init_cfg.clone())
    #     self.assertIsNotNone(Fed_runner)
    #     test_best_results = Fed_runner.run()
    #     print(test_best_results)
    #     init_cfg.merge_from_other_cfg(backup_cfg)
    #     self.assertLess(
    #         test_best_results['client_summarized_weighted_avg']['test_acc'],
    #         0.7)
    #     init_cfg.merge_from_other_cfg(backup_cfg)

    def test_3_bulyan(self):
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
            0.7)
        init_cfg.merge_from_other_cfg(backup_cfg)
    
    # def test_4_normbounding(self):
    #     init_cfg = global_cfg.clone()
    #     backup_cfg = self.set_config_normbounding(init_cfg)
    #     setup_seed(init_cfg.seed)
    #     update_logger(init_cfg, True)

    #     data, modified_cfg = get_data(init_cfg.clone())
    #     init_cfg.merge_from_other_cfg(modified_cfg)
    #     self.assertIsNotNone(data)
    #     Fed_runner = get_runner(data=data,
    #                             server_class=get_server_cls(init_cfg),
    #                             client_class=get_client_cls(init_cfg),
    #                             config=init_cfg.clone())
    #     self.assertIsNotNone(Fed_runner)
    #     test_best_results = Fed_runner.run()
    #     print(test_best_results)
    #     init_cfg.merge_from_other_cfg(backup_cfg)
    #     self.assertLess(
    #         test_best_results['client_summarized_weighted_avg']['test_acc'],
    #         0.7)
    #     init_cfg.merge_from_other_cfg(backup_cfg)

    # def test_5_fltrust(self):
    #     init_cfg = global_cfg.clone()
    #     backup_cfg = self.set_config_fltrust(init_cfg)
    #     setup_seed(init_cfg.seed)
    #     update_logger(init_cfg, True)

    #     data, modified_cfg = get_data(init_cfg.clone())
    #     init_cfg.merge_from_other_cfg(modified_cfg)
    #     self.assertIsNotNone(data)

    #     Fed_runner = get_runner(data=data,
    #                             server_class=get_server_cls(init_cfg),
    #                             client_class=get_client_cls(init_cfg),
    #                             config=init_cfg.clone())
    #     self.assertIsNotNone(Fed_runner)
    #     test_best_results = Fed_runner.run()
    #     print(test_best_results)
    #     init_cfg.merge_from_other_cfg(backup_cfg)
    #     self.assertLess(
    #         test_best_results['client_summarized_weighted_avg']['test_acc'],
    #         0.7)
    #     init_cfg.merge_from_other_cfg(backup_cfg)
    

if __name__ == '__main__':
    unittest.main()
