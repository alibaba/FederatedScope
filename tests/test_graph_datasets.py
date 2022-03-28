# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest

import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

#from flpackage.gfl.dataset import ACM, RecSys, DBLPfull, DBLPNew
from federatedscope.gfl.dataset.splitter import LouvainSplitter, RandomSplitter


class SplitterTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_citation_network(self):
        obs = []
        for num_parts in [3, 5, 8]:
            dataset = Planetoid('test_data',
                                'cora',
                                transform=LouvainSplitter(num_parts))
            obs.append(len(dataset[0]))
        self.assertTrue(np.allclose(obs, [3, 5, 8], atol=1e-3))


if __name__ == '__main__':
    unittest.main()
