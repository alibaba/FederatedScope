# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest

import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

#from federatedscope.gfl.dataset import ACM, RecSys, DBLPfull, DBLPNew
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


'''
if __name__ == "__main__":
    # TODO: change to unittest
    dataset = ACM('data', True, transform=T.NormalizeFeatures())
    print(dataset[0].x.shape, dataset[0].y.shape, dataset[0].y.unique())
    print(dataset.num_features, dataset.num_classes)
    for data in dataset:
        print(data.num_nodes)

    dataset = DBLPfull('data', True, transform=T.NormalizeFeatures())
    print(dataset[0].x.shape, dataset[0].y.shape, dataset[0].y.unique())
    print(dataset.num_features, dataset.num_classes)
    for data in dataset:
        print(data.num_nodes, len(data.y.shape))

    dataset = Planetoid('data', 'cora', transform=LouvainSplitter(3))
    for data in dataset[0]:
        print(data)
        print(data.num_nodes, data.x.shape, data.y.shape, data.y.unique())
        print(data.x.dtype, data.y.dtype, data.index_orig.dtype)

    #dataset = Planetoid('data', 'cora', transform=RandomSplitter(3))
    #print(dataset[0])
    #for data in dataset[0]:
    #    print(data)
    #    print(data.num_nodes, data.x.shape, data.y.shape, data.y.unique())
    #    print(data.x.dtype, data.y.dtype)

    dataset = DBLPNew('data', 0)
    print(len(dataset))
    for data in dataset:
        print(data.num_nodes, data.x.shape, data.y.shape, data.y.unique())

    dataset = DBLPNew('data', 1)
    print(len(dataset))
    for data in dataset:
        print(data)
        print(data.num_nodes, data.x.shape, data.y.shape, data.y.unique())
        print(data.x.dtype, data.y.dtype)

    dataset = DBLPNew('data', 2)
    print(len(dataset))
    for data in dataset:
        print(data)
        print(data.num_nodes, data.x.shape, data.y.shape, data.y.unique())
        print(data.x.dtype, data.y.dtype)
'''

if __name__ == '__main__':
    unittest.main()
