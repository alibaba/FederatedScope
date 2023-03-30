# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import logging
import unittest

from federatedscope.core.configs.config import global_cfg

logger = logging.getLogger(__name__)


class YAMLTest(unittest.TestCase):
    def setUp(self):
        self.exclude_all = ['benchmark', 'scripts', 'federatedscope/autotune']
        self.exclude_file = [
            '.pre-commit-config.yaml', 'meta.yaml',
            'federatedscope/gfl/baseline/isolated_gin_minibatch_on_cikmcup_per_client.yaml',
            'federatedscope/gfl/baseline/fedavg_gin_minibatch_on_cikmcup_per_client.yaml',
            'federatedscope/gfl/baseline/mini_graph_dc/fedavg_per_client.yaml'
        ]
        self.exclude_str = ['config.yaml', 'config_client']
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.exclude_all = [
            os.path.join(self.root, f) for f in self.exclude_all
        ]
        self.exclude_file = [
            os.path.join(self.root, f) for f in self.exclude_file
        ]
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yaml(self):
        init_cfg = global_cfg.clone()
        sign, cont = False, False
        error_file = []
        for dirpath, _, filenames in os.walk(self.root):
            for prefix in self.exclude_all:
                if dirpath.startswith(prefix):
                    cont = True
                    break
            if cont:
                cont = False
                continue
            filenames = [f for f in filenames if f.endswith('.yaml')]
            for f in filenames:
                yaml_file = os.path.join(dirpath, f)
                # Ignore `ss` search space and yaml file in `exclude_file`
                if yaml_file in self.exclude_file or 'ss' in yaml_file:
                    continue
                exclude = False
                for s in self.exclude_str:
                    if s in yaml_file:
                        exclude = True
                        break
                if exclude:
                    continue
                try:
                    init_cfg.merge_from_file(yaml_file)
                except KeyError as error:
                    error_file.append(yaml_file.removeprefix(self.root))
                    logger.error(
                        f"KeyError: {error} in file: {yaml_file.removeprefix(self.root)}"
                    )
                    sign = True
                except ValueError as error:
                    error_file.append(yaml_file.removeprefix(self.root))
                    logger.error(
                        f"ValueError: {error} in file: {yaml_file.removeprefix(self.root)}"
                    )
                    sign = True
                init_cfg = global_cfg.clone()
        self.assertIs(sign, False, f'Yaml check failed in {error_file}.')


if __name__ == '__main__':
    unittest.main()
