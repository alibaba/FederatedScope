import os
import torch
import numpy as np

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import TUDataset, MoleculeNet

from federatedscope.register import register_data
from federatedscope.core.data import DummyDataTranslator
from federatedscope.core.splitters.graph.scaffold_lda_splitter import \
    GenFeatures


class MiniGraphDTDataset(InMemoryDataset):
    NAME = 'mini_graph_dt'
    DATA_NAME = ['ESOL', 'BACE', 'LIPO', 'ENZYMES', 'PROTEINS_full']
    IN_MEMORY_DATA = {}

    def __init__(self, root, splits=[0.8, 0.1, 0.1]):
        self.root = root
        self.splits = splits
        super(MiniGraphDTDataset, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.NAME, 'processed')

    @property
    def processed_file_names(self):
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        return len(self.DATA_NAME)

    def __getitem__(self, idx):
        if idx not in self.IN_MEMORY_DATA:
            self.IN_MEMORY_DATA[idx] = {}
            for split in ['train', 'val', 'test']:
                split_data = self._load(idx, split)
                if split_data:
                    self.IN_MEMORY_DATA[idx][split] = split_data
        return self.IN_MEMORY_DATA[idx]

    def _load(self, idx, split):
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        np.random.seed(0)
        for idx, name in enumerate(self.DATA_NAME):
            if name in ['ESOL', 'BACE', 'LIPO']:
                dataset = MoleculeNet(self.root, name)
                featurizer = GenFeatures()
                ds = []
                for graph in dataset:
                    graph = featurizer(graph)
                    ds.append(
                        Data(edge_index=graph.edge_index, x=graph.x,
                             y=graph.y))
                dataset = ds
                if name in ['ESOL', 'LIPO']:
                    # Regression
                    for i in range(len(dataset)):
                        dataset[i].y = dataset[i].y.squeeze(0)
                if name in ['BACE']:
                    # Classification
                    for i in range(len(dataset)):
                        dataset[i].y = dataset[i].y.long().squeeze(0)
            else:
                # Classification
                dataset = TUDataset(self.root, name)
                dataset = [
                    Data(edge_index=graph.edge_index, x=graph.x, y=graph.y)
                    for graph in dataset
                ]

            # We fix train/val/test
            index = np.random.permutation(np.arange(len(dataset)))
            train_idx = index[:int(len(dataset) * self.splits[0])]
            valid_idx = index[int(len(dataset) * self.splits[0]
                                  ):int(len(dataset) * sum(self.splits[:2]))]
            test_idx = index[int(len(dataset) * sum(self.splits[:2])):]

            if not os.path.isdir(os.path.join(self.processed_dir, str(idx))):
                os.makedirs(os.path.join(self.processed_dir, str(idx)))

            train_path = os.path.join(self.processed_dir, str(idx), 'train.pt')
            valid_path = os.path.join(self.processed_dir, str(idx), 'val.pt')
            test_path = os.path.join(self.processed_dir, str(idx), 'test.pt')

            torch.save([dataset[i] for i in train_idx], train_path)
            torch.save([dataset[i] for i in valid_idx], valid_path)
            torch.save([dataset[i] for i in test_idx], test_path)

    def meta_info(self):
        return {
            'ESOL': {
                'task': 'regression',
                'input_dim': 74,
                'output_dim': 1,
                'num_samples': 1128
            },
            'BACE': {
                'task': 'classification',
                'input_dim': 74,
                'output_dim': 2,
                'num_samples': 1513
            },
            'LIPO': {
                'task': 'regression',
                'input_dim': 74,
                'output_dim': 1,
                'num_samples': 4200
            },
            'ENZYMES': {
                'task': 'classification',
                'input_dim': 3,
                'output_dim': 6,
                'num_samples': 600
            },
            'PROTEINS_full': {
                'task': 'classification',
                'input_dim': 3,
                'output_dim': 2,
                'num_samples': 1113
            },
        }


def load_mini_graph_dt(config, client_cfgs=None):
    dataset = MiniGraphDTDataset(config.data.root)
    # Convert to dict
    datadict = {client_id: dataset[client_id] for client_id in len(dataset)}

    config.merge_from_list(['federate.client_num', len(dataset)])
    translator = DummyDataTranslator(config, client_cfgs)

    return translator(datadict), config


def call_mini_graph_dt(config, client_cfgs):
    if config.data.type == "mini-graph-dt":
        data, modified_config = load_mini_graph_dt(config, client_cfgs)
        return data, modified_config


register_data("mini_graph_dt", call_mini_graph_dt)
