import torch

from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dense_adj, dense_to_sparse


class Analyzer(object):
    r"""Analyzer for raw graph and split subgraphs.

    Arguments:
        raw_data (PyG.data): raw graph.
        split_data (list): the list for subgraphs split by splitter.

    """
    def __init__(self, raw_data: Data, split_data: List[Data]):

        self.raw_data = raw_data
        self.split_data = split_data

        self.raw_graph = to_networkx(raw_data, to_undirected=True)
        self.sub_graphs = [
            to_networkx(g, to_undirected=True) for g in split_data
        ]

    def num_missing_edge(self):
        r"""

        Returns:
            the number of missing edge and the rate of missing edge.

        """
        missing_edge = len(self.raw_graph.edges) - self.fl_adj().shape[1] // 2
        rate_missing_edge = missing_edge / len(self.raw_graph.edges)

        return missing_edge, rate_missing_edge

    def fl_adj(self):
        r"""

        Returns:
            the adj for missing edge ADJ.

        """
        raw_adj = to_dense_adj(self.raw_data.edge_index)[0]
        adj = torch.zeros_like(raw_adj)
        if 'index_orig' in self.split_data[0]:
            for sub_g in self.split_data:
                for row, col in sub_g.edge_index.T:
                    adj[sub_g.index_orig[row.item()]][sub_g.index_orig[
                        col.item()]] = 1

        else:
            raise KeyError('index_orig not in Split Data.')

        return dense_to_sparse(adj)[0]

    def fl_data(self):
        r"""

        Returns:
            the split edge index.

        """
        fl_data = Data()
        for key, item in self.raw_data:
            if key == 'edge_index':
                fl_data[key] = self.fl_adj()
            else:
                fl_data[key] = item

        return fl_data

    def missing_data(self):
        r"""

        Returns:
            the graph data built by missing edge index.

        """
        ms_data = Data()
        raw_edge_set = {tuple(x) for x in self.raw_data.edge_index.T.numpy()}
        split_edge_set = {
            tuple(x)
            for x in self.fl_data().edge_index.T.numpy()
        }
        ms_set = raw_edge_set - split_edge_set
        for key, item in self.raw_data:
            if key == 'edge_index':
                ms_data[key] = torch.tensor([list(x) for x in ms_set],
                                            dtype=torch.int64).T
            else:
                ms_data[key] = item

        return ms_data

    def portion_ms_node(self):
        r"""

        Returns:
            the proportion of nodes who miss egde.

        """
        cnt_list = []
        ms_set = {x.item() for x in set(self.missing_data().edge_index[0])}
        for sub_data in self.split_data:
            cnt = 0
            for idx in sub_data.index_orig:
                if idx.item() in ms_set:
                    cnt += 1
            cnt_list.append(cnt / sub_data.num_nodes)
        return cnt_list

    def average_clustering(self):
        r"""

        Returns:
            the average clustering coefficient for the raw G and split G

        """
        import networkx.algorithms.cluster as cluster

        return cluster.average_clustering(
            self.raw_graph), cluster.average_clustering(
                to_networkx(self.fl_data()))

    def homophily_value(self, edge_index, y):
        r"""

        Returns:
            calculate homophily_value

        """
        from torch_sparse import SparseTensor

        if isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
        else:
            row, col = edge_index

        return int((y[row] == y[col]).sum()) / row.size(0)

    def homophily(self):
        r"""

        Returns:
            the homophily for the raw G and split G

        """

        return self.homophily_value(self.raw_data.edge_index,
                                    self.raw_data.y), self.homophily_value(
                                        self.fl_data().edge_index,
                                        self.fl_data().y)

    def hamming_distance_graph(self, data):
        r"""

        Returns:
            calculate the hamming distance of graph data

        """
        edge_index, x = data.edge_index, data.x
        cnt = 0
        for row, col in edge_index.T:
            row, col = row.item(), col.item()
            cnt += torch.sum(x[row] != x[col]).item()

        return cnt / edge_index.shape[1]

    def hamming(self):
        r"""

        Returns:
            the average hamming distance of feature for the raw G, split G
            and missing edge G

        """
        return self.hamming_distance_graph(
            self.raw_data), self.hamming_distance_graph(
                self.fl_data()), self.hamming_distance_graph(
                    self.missing_data())
