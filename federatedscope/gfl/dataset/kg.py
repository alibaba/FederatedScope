"""This file is part of https://github.com/pyg-team/pytorch_geometric
Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de,
jiaxuan@cs.stanford.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url


class KG(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
        return [
            'entities.dict', 'relations.dict', 'test.txt', 'train.txt',
            'valid.txt'
        ]

    def download(self):
        url = 'https://github.com/MichSchli/RelationPrediction/tree/master' \
              '/data/'
        urls = {
            'fb15k': url + 'FB15k',
            'fb15k-237': url + 'FB-Toutanova',
            'wn18': url + 'wn18',
            'toy': url + 'Toy'
        }
        for file_name in self.raw_file_names:
            download_url(f'{urls[self.name]}/{file_name}', self.raw_dir)

    def process(self):
        with open(osp.join(self.raw_dir, 'entities.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(osp.join(self.raw_dir, 'relations.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        kwargs = {}
        for split in ['train', 'valid', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.txt'), 'r') as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                kwargs[f'{split}_edge_index'] = torch.tensor([src, dst])
                kwargs[f'{split}_edge_type'] = torch.tensor(rel)

        # For message passing, we add reverse edges and types to the graph:
        row, col = kwargs['train_edge_index']
        edge_type = kwargs['train_edge_type']
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type + len(relations_dict)])
        num_nodes = len(entities_dict)
        data = Data(num_nodes=num_nodes,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    **kwargs)
        edge_index = torch.cat((data.train_edge_index, data.valid_edge_index,
                                data.test_edge_index),
                               dim=-1)
        edge_type = torch.cat(
            (data.train_edge_type, data.valid_edge_type, data.test_edge_type),
            dim=0)
        num_edges = edge_index.size(-1)
        train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_edge_mask[:data.train_edge_index.size(-1)] = True
        valid_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        valid_edge_mask[data.train_edge_index.
                        size(-1):-data.test_edge_index.size(-1)] = True
        test_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_edge_mask[-data.test_edge_index.size(-1):] = True
        data = Data(edge_index=edge_index,
                    index_orig=torch.arange(num_nodes),
                    edge_type=edge_type,
                    num_nodes=num_nodes,
                    train_edge_mask=train_edge_mask,
                    valid_edge_mask=valid_edge_mask,
                    test_edge_mask=test_edge_mask,
                    input_edge_index=data.edge_index)

        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save((self.collate([data])), self.processed_paths[0])
