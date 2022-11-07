# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, \
    get_client_cls
from federatedscope.core.configs.yacs_config import CfgNode


class NLPTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_isolated(self):
        cfg_alg = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_isolated.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_client_isolated.yaml',
                 'r'))
        cfg_alg.outdir = 'exp/fedavg/'
        self.fed_runner(cfg_alg, cfg_client)

    def test_fednlp(self):
        cfg_alg = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_fednlp.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_client_fednlp.yaml', 'r'))
        cfg_alg.outdir = 'exp/fednlp/'
        self.fed_runner(cfg_alg, cfg_client)

    def test_pfednlp(self):
        # pretrain
        cfg_alg = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_pfednlp_pretrain.yaml',
                 'r'))
        cfg_alg.outdir = 'exp/pfednlp/pretrain/'
        self.fed_runner(cfg_alg)

        # train
        cfg_alg = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_pfednlp.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_client_pfednlp.yaml',
                 'r'))
        cfg_alg.outdir = 'exp/pfednlp/train/'
        self.fed_runner(cfg_alg, cfg_client)

    def test_pcfednlp(self):
        cfg_alg = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_pcfednlp.yaml', 'r'))
        cfg_client = CfgNode.load_cfg(
            open('federatedscope/nlp/baseline/config_client_pcfednlp.yaml',
                 'r'))
        cfg_alg.outdir = 'exp/pcfednlp/'
        self.fed_runner(cfg_alg, cfg_client)

    def fed_runner(self, cfg_alg, cfg_client=None):
        init_cfg = global_cfg.clone()
        init_cfg.merge_from_other_cfg(cfg_alg)
        init_cfg.data.debug = True
        init_cfg.data.batch_size = 1
        init_cfg.data.root = 'test_data/'
        init_cfg.data.datasets = ['imdb', 'agnews', 'squad', 'newsqa']
        init_cfg.data.num_grouped_clients = [1, 3, 3, 2]
        init_cfg.federate.client_num = 9
        setup_seed(init_cfg.seed)
        update_logger(init_cfg, True)

        data, modified_cfg = get_data(init_cfg.clone(), client_cfgs=cfg_client)
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        if cfg_client is not None:
            num_client = init_cfg.federate.client_num
            for i in range(1, num_client + 1):
                cfg = cfg_client['client_{}'.format(i)]
                cfg.data.batch_size = 1

        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(init_cfg),
                               client_class=get_client_cls(init_cfg),
                               config=init_cfg.clone(),
                               client_configs=cfg_client)
        self.assertIsNotNone(Fed_runner)
        _ = Fed_runner.run()

        if 'Results_group_avg' in Fed_runner.server.history_results:
            test_res = Fed_runner.server.history_results['Results_group_avg']
            eval_res = [x['test_avg_loss'][-1] for x in test_res.values()]
            eval_res = sum(eval_res) / len(eval_res)
            print('Eval results: {}, test results: {}'.format(
                eval_res, test_res))
            self.assertLess(eval_res, 50)


if __name__ == '__main__':
    unittest.main()
