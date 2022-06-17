# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
import os.path as osp

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls

from yacs.config import CfgNode


class TextDTTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_fedavg_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_fedavg.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_fedavg.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/fedavg'
        cfg_alg.federate.save_to = osp.join(cfg_alg.outdir, 'ckpt/global_model.pt')
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def test_fedavg_ft_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_ft.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_fedavg_ft.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/fedavg_ft'
        cfg_alg.federate.method = 'fedavg-textdt'
        cfg_alg.federate.load_from = 'exp_out/fedavg/ckpt'
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def test_ditto_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_ditto.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_ditto.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/ditto'
        cfg_alg.federate.save_to = osp.join(cfg_alg.outdir, 'ckpt/global_model.pt')
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def test_fedbn_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_fedbn.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_fedbn.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/fedbn'
        cfg_alg.federate.save_to = osp.join(cfg_alg.outdir, 'ckpt/global_model.pt')
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def test_fedbn_ft_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_ft.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_fedbn_ft.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/fedbn_ft'
        cfg_alg.federate.method = 'fedbn-textdt'
        cfg_alg.federate.load_from = 'exp_out/fedbn/ckpt'
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def test_fedprox_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_fedprox.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_fedprox.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/fedprox'
        cfg_alg.federate.save_to = osp.join(cfg_alg.outdir, 'ckpt/global_model.pt')
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def test_fedmaml_standalone(self):
        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_maml.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_maml.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/maml/train'
        cfg_alg.federate.save_to = osp.join(cfg_alg.outdir, 'ckpt/global_model.pt')
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

        cfg_alg = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_ft.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(open('scripts/B-FHTL_exp_scripts/Text-DT/config_client_maml_ft.yaml', 'r'))
        cfg_alg.outdir = 'exp_out/maml/ft'
        cfg_alg.federate.method = 'maml-textdt'
        cfg_alg.federate.load_from = 'exp_out/maml/train/ckpt'
        self.fedrunner(
            cfg_alg=cfg_alg,
            cfg_client=cfg_client,
        )

    def fedrunner(self, cfg_alg, cfg_client):
        init_cfg = global_cfg.clone()
        init_cfg.merge_from_other_cfg(cfg_alg)
        init_cfg.data.root = 'test_data/'
        init_cfg.data.dir.sts = 'test_data/STS-B/'
        init_cfg.data.dir.imdb = 'test_data/imdb/'
        init_cfg.data.dir.squad = 'test_data/squad2.0/'
        init_cfg.data.cache_dir = 'test_data/cache_debug/'
        init_cfg.data.batch_size = 2
        init_cfg.data.debug = True
        if init_cfg.federate.total_round_num > 1:
            init_cfg.federate.total_round_num = 2
        setup_seed(init_cfg.seed)
        update_logger(init_cfg)

        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        num_client = init_cfg.federate.client_num
        for i in range(1, num_client + 1):
            cfg = cfg_client['client_{}'.format(i)]
            cfg.trainer.train_steps = 5

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(init_cfg),
                               client_class=get_client_cls(init_cfg),
                               config=init_cfg.clone(),
                               config_client=cfg_client.clone())
        self.assertIsNotNone(Fed_runner)
        test_best_results = Fed_runner.run()
        print('Test best results:')
        print(test_best_results)
        self.assertLess(test_best_results["client_individual"]['test_avg_loss'], 50)


if __name__ == '__main__':
    unittest.main()
