import random
import numpy as np

from federatedscope.core.splitters import BaseSplitter


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self, client_num, **kwargs):
        super(MetaSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')

        # Split by categories
        categories = set(label)
        idx_slice = []
        for cat in categories:
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())
        random.shuffle(idx_slice)

        # Merge to client_num pieces
        new_idx_slice = []
        for i in range(len(categories)):
            if i < self.client_num:
                new_idx_slice.append(idx_slice[i])
            else:
                new_idx_slice[i % self.client_num] += idx_slice[i]

        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
