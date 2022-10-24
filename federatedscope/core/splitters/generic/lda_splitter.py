import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


class LDASplitter(BaseSplitter):
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        dataset = [ds for ds in dataset]
        label = np.array([y for x, y in dataset])
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list
