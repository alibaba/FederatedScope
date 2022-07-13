# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from federatedscope.core.configs.config import global_cfg


class YAMLTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yaml(self):
        init_cfg = global_cfg.clone()
        for dirpath, _, filenames in os.walk('../federatedscope'):
            filenames = [f for f in filenames if f.endswith('.yaml')]
            for f in filenames:
                file = os.path.join(dirpath, f)
                init_cfg.merge_from_file(file)


if __name__ == '__main__':
    unittest.main()
