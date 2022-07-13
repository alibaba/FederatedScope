# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class FedSagePlusTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_fedsageplus(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()

        cfg.federate.mode = 'standalone'
        cfg.federate.make_global_eval = True
        cfg.federate.client_num = 3
        cfg.federate.total_round_num = 10
        cfg.federate.method = 'fedsageplus'
        cfg.train.batch_or_epoch = 'epoch'

        cfg.data.root = 'test_data/'
        cfg.data.type = 'cora'
        cfg.data.splitter = 'louvain'
        cfg.data.batch_size = 1

        cfg.model.type = 'sage'
        cfg.model.hidden = 64
        cfg.model.dropout = 0.5
        cfg.model.out_channels = 7

        cfg.fedsageplus.num_pred = 5
        cfg.fedsageplus.gen_hidden = 64
        cfg.fedsageplus.hide_portion = 0.5
        cfg.fedsageplus.fedgen_epoch = 2
        cfg.fedsageplus.loc_epoch = 1
        cfg.fedsageplus.a = 1.0
        cfg.fedsageplus.b = 1.0
        cfg.fedsageplus.c = 1.0

        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'nodefullbatch_trainer'
        cfg.eval.metrics = ['acc', 'correct']

        return backup_cfg

    def test_fedsageplus_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_fedsageplus(init_cfg)
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
        init_cfg.merge_from_other_cfg(backup_cfg)
        self.assertGreater(test_best_results["server_global_eval"]['test_acc'],
                           0.7)


if __name__ == '__main__':
    unittest.main()
