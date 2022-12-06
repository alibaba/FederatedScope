import os
import numpy as np

from federatedscope.register import register_data

# Run with mini_graph_dt:
# python federatedscope/main.py --cfg \
# federatedscope/gfl/baseline/mini_graph_dc/fedavg.yaml --client_cfg \
# federatedscope/gfl/baseline/mini_graph_dc/fedavg_per_client.yaml
# Test Accuracy: ~0.7


def load_mini_graph_dt(config, client_cfgs=None):
    import torch
    from torch_geometric.data import InMemoryDataset, Data
    from torch_geometric.datasets import TUDataset, MoleculeNet
    from federatedscope.core.splitters.graph.scaffold_lda_splitter import \
        GenFeatures
    from federatedscope.core.data import DummyDataTranslator

    class MiniGraphDCDataset(InMemoryDataset):
        NAME = 'mini_graph_dt'
        DATA_NAME = ['BACE', 'BBBP', 'CLINTOX', 'ENZYMES', 'PROTEINS_full']
        IN_MEMORY_DATA = {}

        def __init__(self, root, splits=[0.8, 0.1, 0.1]):
            self.root = root
            self.splits = splits
            super(MiniGraphDCDataset, self).__init__(root)

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
                if name in ['BACE', 'BBBP', 'CLINTOX']:
                    dataset = MoleculeNet(self.root, name)
                    featurizer = GenFeatures()
                    ds = []
                    for graph in dataset:
                        graph = featurizer(graph)
                        ds.append(
                            Data(edge_index=graph.edge_index,
                                 x=graph.x,
                                 y=graph.y))
                    dataset = ds
                    if name in ['BACE', 'BBBP']:
                        for i in range(len(dataset)):
                            dataset[i].y = dataset[i].y.long()
                    if name in ['CLINTOX']:
                        for i in range(len(dataset)):
                            dataset[i].y = torch.argmax(
                                dataset[i].y).view(-1).unsqueeze(0)
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
                valid_idx = index[int(len(dataset) * self.splits[0]):int(
                    len(dataset) * sum(self.splits[:2]))]
                test_idx = index[int(len(dataset) * sum(self.splits[:2])):]

                if not os.path.isdir(os.path.join(self.processed_dir,
                                                  str(idx))):
                    os.makedirs(os.path.join(self.processed_dir, str(idx)))

                train_path = os.path.join(self.processed_dir, str(idx),
                                          'train.pt')
                valid_path = os.path.join(self.processed_dir, str(idx),
                                          'val.pt')
                test_path = os.path.join(self.processed_dir, str(idx),
                                         'test.pt')

                torch.save([dataset[i] for i in train_idx], train_path)
                torch.save([dataset[i] for i in valid_idx], valid_path)
                torch.save([dataset[i] for i in test_idx], test_path)

                print(name, len(dataset), dataset[0])

        def meta_info(self):
            return {
                'BACE': {
                    'task': 'classification',
                    'input_dim': 74,
                    'output_dim': 2,
                    'num_samples': 1513,
                },
                'BBBP': {
                    'task': 'classification',
                    'input_dim': 74,
                    'output_dim': 2,
                    'num_samples': 2039,
                },
                'CLINTOX': {
                    'task': 'classification',
                    'input_dim': 74,
                    'output_dim': 2,
                    'num_samples': 1478,
                },
                'ENZYMES': {
                    'task': 'classification',
                    'input_dim': 3,
                    'output_dim': 6,
                    'num_samples': 600,
                },
                'PROTEINS_full': {
                    'task': 'classification',
                    'input_dim': 3,
                    'output_dim': 2,
                    'num_samples': 1113,
                },
            }

    dataset = MiniGraphDCDataset(config.data.root)
    # Convert to dict
    datadict = {
        client_id + 1: dataset[client_id]
        for client_id in range(len(dataset))
    }
    config.merge_from_list(['federate.client_num', len(dataset)])
    translator = DummyDataTranslator(config, client_cfgs)

    return translator(datadict), config


def call_mini_graph_dt(config, client_cfgs):
    if config.data.type == "mini-graph-dc":
        data, modified_config = load_mini_graph_dt(config, client_cfgs)
        return data, modified_config


register_data("mini-graph-dc", call_mini_graph_dt)
