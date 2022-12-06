import logging
import federatedscope.register as register
from federatedscope.nlp.scheduler import *

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    from federatedscope.contrib.scheduler import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.scheduler`, some modules are not '
        f'available.')


def get_scheduler(optimizer, type, **kwargs):
    """
    This function builds an instance of scheduler.

    Args:
        optimizer: optimizer to be scheduled
        type: type of scheduler
        **kwargs: kwargs dict

    Returns:
        An instantiated scheduler.

    Note:
        Please follow ``contrib.scheduler.example`` to implement your own \
        scheduler.
    """
    for func in register.scheduler_dict.values():
        scheduler = func(optimizer, type, **kwargs)
        if scheduler is not None:
            return scheduler

    if torch is None or type == '':
        return None
    if isinstance(type, str):
        if hasattr(torch.optim.lr_scheduler, type):
            return getattr(torch.optim.lr_scheduler, type)(optimizer, **kwargs)
        else:
            raise NotImplementedError(
                'Scheduler {} not implement'.format(type))
    else:
        raise TypeError()
