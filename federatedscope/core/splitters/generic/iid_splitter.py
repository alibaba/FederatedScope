import numpy as np
from federatedscope.core.splitters import BaseSplitter


class IIDSplitter(BaseSplitter):
    """
    This splitter splits dataset following the independent and identically \
    distribution.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num):
        super(IIDSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None):
        from torch.utils.data import Dataset, Subset

        length = len(dataset)
        index = [x for x in range(length)]
        np.random.shuffle(index)
        idx_slice = np.array_split(np.array(index), self.client_num)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
