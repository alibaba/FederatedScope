import federatedscope.register as register

try:
    from torch import nn
    from federatedscope.nlp.loss import *
except ImportError:
    nn = None


def get_criterion(type, device):
    for func in register.criterion_dict.values():
        criterion = func(type, device)
        if criterion is not None:
            return criterion

    if isinstance(type, str):
        if hasattr(nn, type):
            return getattr(nn, type)()
        else:
            raise NotImplementedError(
                'Criterion {} not implement'.format(type))
    else:
        raise TypeError()
