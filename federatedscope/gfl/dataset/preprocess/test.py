from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyData(Dataset):
    def __init__(self):
        self.data = np.random.normal(0, 0.1, size=[4, 2])
        # self.label = np.random.normal(0, 0.2, size=[4])
        self.label = np.random.randint(0, 10, size=[4])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

if __name__ == '__main__':
    loader = DataLoader(MyData(), batch_size=2)

    for data, label in loader:
        print(data, data.dtype, label, label.dtype)