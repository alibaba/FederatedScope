import numpy as np

from torch_geometric.transforms import BaseTransform
from federatedscope.core.splitters import BaseSplitter


class RandChunkSplitter(BaseTransform, BaseSplitter):
    """
    Split graph-level dataset via random chunk strategy.

    Arguments:
        dataset (List or PyG.dataset): The graph-level datasets.
    """
    def __init__(self, client_num):
        BaseSplitter.__init__(self, client_num)

    def __call__(self, dataset, **kwargs):
        data_list = []
        dataset = [ds for ds in dataset]
        num_graph = len(dataset)

        # Split dataset
        num_graph = len(dataset)
        min_size = min(50, int(num_graph / self.client_num))

        for i in range(self.client_num):
            data_list.append(dataset[i * min_size:(i + 1) * min_size])
        for graph in dataset[self.client_num * min_size:]:
            client_idx = np.random.randint(low=0, high=self.client_num,
                                           size=1)[0]
            data_list[client_idx].append(graph)

        return data_list
