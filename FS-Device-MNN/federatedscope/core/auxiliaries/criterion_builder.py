import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from torch import nn
    from federatedscope.nlp.loss import *
except ImportError:
    nn = None

try:
    from federatedscope.contrib.loss import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.loss`, some modules are not '
        f'available.')


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
