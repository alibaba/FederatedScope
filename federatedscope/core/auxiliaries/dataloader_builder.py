import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_dataloader(dataset, config):
    dataloader = DataLoader(dataset,
                            batch_size=config.data.batch_size,
                            shuffle=config.data.shuffle,
                            num_workers=config.data.num_workers,
                            pin_memory=True)
    return dataloader


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
