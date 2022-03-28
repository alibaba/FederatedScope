import torch
from federatedscope.gfl.dataset.utils import dirichlet_distribution_noniid_slice


class GraphTypeSplitter:
    def __init__(self, client_num, alpha):
        self.client_num = client_num
        self.alpha = alpha

    def __call__(self, dataset):
        r"""Split dataset via dirichlet distribution to generate non-i.i.d data split.
        
        Arguments:
            dataset (List or PyG.dataset): The datasets.
            
        Returns:
            data_list (List(List(PyG.data))): Splited dataset via dirichlet.
        """
        data_list = []
        dataset = [ds for ds in dataset]
        label = torch.LongTensor([ds.y.item() for ds in dataset])
        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}()'