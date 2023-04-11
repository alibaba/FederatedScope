try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def get_dataloader(dataset, config):
    if config.backend == 'torch':
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset,
                                batch_size=config.data.batch_size,
                                shuffle=config.data.shuffle,
                                num_workers=config.data.num_workers,
                                pin_memory=True)
        return dataloader
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
