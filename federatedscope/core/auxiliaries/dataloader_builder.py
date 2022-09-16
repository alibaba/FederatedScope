from federatedscope.core.data.utils import get_func_args, filter_dict

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def get_dataloader(dataset, config, split='train'):
    if config.backend == 'torch':
        if config.dataloader.type == 'base':
            from torch.utils.data import DataLoader
            loader_cls = DataLoader
        elif config.dataloader.type == 'raw':
            loader_cls = None
        elif config.dataloader.type == 'pyg':
            from torch_geometric.loader import DataLoader as PyGDataLoader
            loader_cls = PyGDataLoader
        elif config.dataloader.type == 'graphsaint-rw':
            if split == 'train':
                from torch_geometric.loader import GraphSAINTRandomWalkSampler
                loader_cls = GraphSAINTRandomWalkSampler
            else:
                from torch_geometric.loader import NeighborSampler
                loader_cls = NeighborSampler
        elif config.dataloader.type == 'neighbor':
            from torch_geometric.loader import NeighborSampler
            loader_cls = NeighborSampler
        elif config.dataloader.type == 'mf':
            from federatedscope.mf.dataloader import MFDataLoader
            loader_cls = MFDataLoader
        else:
            raise ValueError(f'data.loader.type {config.data.loader.type} '
                             f'not found!')
        if loader_cls is not None:
            raw_args = dict(config.dataloader)
            if split != 'train':
                raw_args['shuffle'] = False
                raw_args['sizes'] = [-1]
                # For evaluation in GFL
                if config.dataloader.type in ['graphsaint-rw', 'neighbor']:
                    raw_args['batch_size'] = 4096
                    dataset = dataset[0].edge_index
            else:
                if config.dataloader.type in ['graphsaint-rw']:
                    # Raw graph
                    dataset = dataset[0]
                elif config.dataloader.type in ['neighbor']:
                    # edge_index of raw graph
                    dataset = dataset[0].edge_index
            filtered_args = filter_dict(loader_cls.__init__, raw_args)
            dataloader = loader_cls(dataset, **filtered_args)
            return dataloader
        else:
            return dataset
    else:
        return None


class WrapDataset(Dataset):
    """Wrap raw data into pytorch Dataset

    Arguments:
        data (dict): raw data dictionary contains "x" and "y"

    """
    def __init__(self, data):
        super(WrapDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        if not isinstance(self.data["x"][idx], torch.Tensor):
            return torch.from_numpy(
                self.data["x"][idx]).float(), torch.from_numpy(
                    self.data["y"][idx]).float()
        return self.data["x"][idx], self.data["y"][idx]

    def __len__(self):
        return len(self.data["y"])
