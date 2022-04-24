# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class FedExTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_toy_fedex(self):
        case_cfg = global_cfg.clone()
        case_cfg.merge_from_file(
            'federatedscope/example_configs/fedex_for_lr.yaml')

        setup_seed(case_cfg.seed)
        update_logger(case_cfg)

        data, _ = get_data(case_cfg.clone())
        Fed_runner = FedRunner(data=data,
                               server_class=get_server_cls(case_cfg),
                               client_class=get_client_cls(case_cfg),
                               config=case_cfg.clone())
        results = Fed_runner.run()

        self.assertLess(results["client_summarized_weighted_avg"]['test_avg_loss'],
                        0.3)


if __name__ == '__main__':
    unittest.main()
