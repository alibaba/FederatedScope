import logging
import federatedscope.register as register

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


def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    This function builds an instance of scheduler.

    Args:
        optimizer: optimizer to be scheduled
        scheduler_type: type of scheduler
        **kwargs: kwargs dict

    Returns:
        An instantiated scheduler.

    Note:
        Please follow ``contrib.scheduler.example`` to implement your own \
        scheduler.
    """
    for func in register.scheduler_dict.values():
        scheduler = func(optimizer, scheduler_type)
        if scheduler is not None:
            return scheduler

    if torch is None or scheduler_type == '':
        return None
    if isinstance(scheduler_type, str):
        if hasattr(torch.optim.lr_scheduler, scheduler_type):
            return getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer,
                                                                     **kwargs)
        else:
            raise NotImplementedError(
                'Scheduler {} not implement'.format(scheduler_type))
    else:
        raise TypeError()
