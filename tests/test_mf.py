# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class MFTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_movielens1m(self, cfg):
        backup_cfg = cfg.clone()

        import torch
        cfg.use_gpu = torch.cuda.is_available()
        cfg.early_stop_patience = 100
        cfg.eval.best_res_update_round_wise_key = "test_avg_loss"
        cfg.eval.freq = 5
        cfg.eval.metrics = []

        cfg.federate.mode = 'standalone'
        cfg.train.local_update_steps = 20
        cfg.federate.total_round_num = 50
        cfg.federate.client_num = 5

        cfg.data.root = 'test_data/'
        cfg.data.type = 'vflmovielens1m'
        cfg.data.batch_size = 32

        cfg.model.type = 'VMFNet'
        cfg.model.hidden = 20

        cfg.train.optimizer.lr = 1.
        cfg.train.optimizer.weight_decay = 0.0

        cfg.criterion.type = 'MSELoss'
        cfg.trainer.type = 'mftrainer'
        cfg.seed = 123

        return backup_cfg

    def test_mf_standalone(self):
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_movielens1m(init_cfg)
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
        test_results = Fed_runner.run()
        init_cfg.merge_from_other_cfg(backup_cfg)

        self.assertLess(
            test_results["client_summarized_weighted_avg"]["test_avg_loss"],
            50)


if __name__ == '__main__':
    unittest.main()
