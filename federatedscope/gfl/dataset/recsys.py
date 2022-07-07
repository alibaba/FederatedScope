import os

import numpy as np
import os.path as osp
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_networkx

from federatedscope.gfl.dataset.utils import random_planetoid_splits


# RecSys
def read_mapping(path, filename):
    mapping = {}
    with open(os.path.join(path, filename)) as f:
        for line in f:
            s = line.strip().split()
            mapping[int(s[0])] = int(s[1])

    return mapping


def partition_by_category(graph, mapping_item2category):
    partition = {}
    for key in mapping_item2category:
        partition[key] = [mapping_item2category[key]]
        for neighbor in graph.neighbors(key):
            if neighbor not in partition:
                partition[neighbor] = []
            partition[neighbor].append(mapping_item2category[key])
    return partition


def subgraphing(g, partion, mapping_item2category):
    nodelist = [[] for i in set(mapping_item2category.keys())]
    for k, v in partion.items():
        for category in v:
            nodelist[category].append(k)

    graphs = []
    for nodes in nodelist:
        if len(nodes) < 2:
            continue
        graph = nx.subgraph(g, nodes)
        graphs.append(from_networkx(graph))
    return graphs


def read_RecSys(path, FL=False):
    mapping_user = read_mapping(path, 'user.dict')
    mapping_item = read_mapping(path, 'item.dict')

    G = nx.Graph()
    with open(osp.join(path, 'graph.txt')) as f:
        for line in f:
            s = line.strip().split()
            s = [int(i) for i in s]
            G.add_edge(mapping_user[s[0]], mapping_item[s[1]], edge_type=s[2])
    dic = {}
    for node in G.nodes:
        dic[node] = node
    nx.set_node_attributes(G, dic, "index_orig")
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    G = H
    if FL:
        mapping_item2category = read_mapping(path, "category.dict")
        partition = partition_by_category(G, mapping_item2category)
        graphs = subgraphing(G, partition, mapping_item2category)
        return graphs
    else:
        return [from_networkx(G)]


class RecSys(InMemoryDataset):
    r"""
    Arguments:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"epinions"`,
        :obj:`"ciao"`).
        FL (Bool): Federated setting or centralized setting.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self,
                 root,
                 name,
                 FL=False,
                 splits=[0.8, 0.1, 0.1],
                 transform=None,
                 pre_transform=None):
        self.FL = FL
        if self.FL:
            self.name = 'FL' + name
        else:
            self.name = name
        self._customized_splits = splits
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['user.dict', 'item.dict', 'category.dict', 'graph.txt']
        return names

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    def download(self):
        # Download to `self.raw_dir`.
        url = 'https://github.com/FedML-AI/FedGraphNN/tree/main/data' \
              '/recommender_system'
        if self.name.startswith('FL'):
            suffix = self.name[2:]
        else:
            suffix = self.name
        url = osp.join(url, suffix)
        for name in self.raw_file_names:
            download_url(f'{url}/{name}', self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_RecSys(self.raw_dir, self.FL)

        data_list_w_masks = []
        for data in data_list:
            if self.name.endswith('epinions'):
                data.edge_type = data.edge_type - 1
            if data.num_edges == 0:
                continue
            indices = torch.randperm(data.num_edges)
            data.train_edge_mask = torch.zeros(data.num_edges,
                                               dtype=torch.bool)
            data.train_edge_mask[indices[:round(self._customized_splits[0] *
                                                data.num_edges)]] = True
            data.valid_edge_mask = torch.zeros(data.num_edges,
                                               dtype=torch.bool)
            data.valid_edge_mask[indices[
                round(self._customized_splits[0] *
                      data.num_edges):round((self._customized_splits[0] +
                                             self._customized_splits[1]) *
                                            data.num_edges)]] = True
            data.test_edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
            data.test_edge_mask[indices[round((self._customized_splits[0] +
                                               self._customized_splits[1]) *
                                              data.num_edges):]] = True
            data_list_w_masks.append(data)
        data_list = data_list_w_masks

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
