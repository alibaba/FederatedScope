# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls

from yacs.config import CfgNode


class GraphDTTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_fedavg_standalone(self):
        self.fedrunner(
            cfg_alg='../scripts/B-FHTL_exp_scripts/Graph-DT/hpo/fedavg_gnn_minibatch_on_multi_task.yaml',
            cfg_client='../scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'
        )
    #
    # def test_fedavg_ft_standalone(self):
    #     self.fedrunner(
    #         cfg_alg='../scripts/B-FHTL_exp_scripts/Graph-DT/fedavg_gnn_minibatch_on_multi_task.yaml',
    #         cfg_client='../scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'
    #     )
    #
    # def test_ditto_standalone(self):
    #     self.fedrunner(
    #         cfg_alg='../scripts/B-FHTL_exp_scripts/Graph-DT/fedavg_gnn_minibatch_on_multi_task.yaml',
    #         cfg_client='../scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'
    #     )
    #
    # def test_fedbn_standalone(self):
    #     cfg_alg = CfgNode.load_cfg(open("../scripts/B-FHTL_exp_scripts/Graph-DT/fedbn.yaml", 'r'))
    #     self.fedrunner(cfg_alg)
    #
    # def test_fedbn_ft_standalone(self):
    #     cfg_alg = CfgNode.load_cfg(open("../scripts/B-FHTL_exp_scripts/Graph-DT/fedbn_ft.yaml", 'r'))
    #     self.fedrunner(cfg_alg)
    #
    # def test_fedprox_standalone(self):
    #     cfg_alg = CfgNode.load_cfg(open("../scripts/B-FHTL_exp_scripts/Graph-DT/fedprox.yaml", 'r'))
    #     self.fedrunner(cfg_alg)
    #
    # def test_fedmaml_standalone(self):
    #     cfg_alg = CfgNode.load_cfg(open("../scripts/B-FHTL_exp_scripts/Graph-DT/fedmaml.yaml", 'r'))
    #     self.fedrunner(cfg_alg)

    def fedrunner(self, cfg_alg, cfg_client):
        cfg_alg = CfgNode.load_cfg(open(cfg_alg, 'r'))
        client_cfg = CfgNode.load_cfg(open(cfg_client, 'r'))

        init_cfg = global_cfg.clone()
        init_cfg.merge_from_other_cfg(cfg_alg)
        init_cfg.federate.total_round_num = 10
        init_cfg.eval.freq = 10
        init_cfg.data.root = 'test_data/'
        setup_seed(init_cfg.seed)
        update_logger(init_cfg)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(init_cfg),
                               client_class=get_client_cls(init_cfg),
                               config=init_cfg.clone(),
                               config_client=client_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print(test_best_results)
        self.assertLess(
            test_best_results["client_summarized_weighted_avg"]['test_loss'],
            500)


if __name__ == '__main__':
    unittest.main()