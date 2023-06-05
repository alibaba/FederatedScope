import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
