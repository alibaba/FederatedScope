import torch
import federatedscope.register as register

from federatedscope.nlp.loss import *


def get_criterion(type, device):
    for func in register.criterion_dict.values():
        criterion = func(type, device)
        if criterion is not None:
            return criterion

    if isinstance(type, str):
        if hasattr(torch.nn, type):
            return getattr(torch.nn, type)()
        else:
            raise NotImplementedError(
                'Criterion {} not implement'.format(type))
    else:
        raise TypeError()
