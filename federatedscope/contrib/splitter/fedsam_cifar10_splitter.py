'''The implementation of ASAM and SAM are borrowed from
    https://github.com/debcaldarola/fedsam
   Caldarola, D., Caputo, B., & Ciccone, M.
   Improving Generalization in Federated Learning by Seeking Flat Minima,
   European Conference on Computer Vision (ECCV) 2022.
'''
import re
import json
import numpy as np

from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class FedSAM_CIFAR10_Splitter(BaseSplitter):
    """
    This splitter split according to what FedSAM provides

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(FedSAM_CIFAR10_Splitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        dataset = [ds for ds in dataset]
        label = np.array([y for x, y in dataset])
        alpha_str = f'{self.alpha:.2f}'
        if len(label) == 50000:
            filename = \
                'data/fedsam_cifar10/data/{}/federated_{}_alpha_{' \
                '}.json'.format('train', 'train', alpha_str)
        elif len(label) == 10000:
            filename = 'data/fedsam_cifar10/data/test/test.json'
        with open(filename, 'r') as ips:
            content = json.load(ips)
            idx_slice = []

            def get_idx(name_list):
                return [
                    int(re.findall('img_\d+_label', fn)[0][4:-6])
                    for fn in name_list
                ]

            if len(label) == 50000:
                for uid in range(self.client_num):
                    idx_slice.append(
                        get_idx(content['user_data'][str(uid)]['x']))
            elif len(label) == 10000:
                idx_slice.append(get_idx(content['user_data'][str(100)]['x']))
                idx_slice = np.array_split(np.array(idx_slice[0]),
                                           self.client_num)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list


def call_fedsam_cifar10_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'fedsam_cifar10_splitter':
        splitter = FedSAM_CIFAR10_Splitter(client_num, **kwargs)
        return splitter


register_splitter('fedsam_cifar10_splitter', call_fedsam_cifar10_splitter)
