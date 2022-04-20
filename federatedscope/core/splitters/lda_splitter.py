import torch
from federatedscope.core.splitters.utils import dirichlet_distribution_noniid_slice


class LDASplitter(object):
    def __init__(self, client_num, alpha):
        self.client_num = client_num
        self.alpha = alpha

    def __call__(self, dataset):
        dataset = [ds for ds in dataset]
        label = torch.LongTensor([y for x, y in dataset])
        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_client}, {self.alpha})'