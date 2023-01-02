import logging
import federatedscope.register as register
from federatedscope.nlp.hetero_tasks.scheduler import *

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

    if type == 'warmup_step':
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = kwargs['warmup_steps']
        total_steps = kwargs['total_steps']

        def lr_lambda(cur_step):
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - cur_step) /
                float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)
    elif type == 'warmup_noam':
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = kwargs['warmup_steps']

        def lr_lambda(cur_step):
            return min(
                max(1, cur_step)**(-0.5),
                max(1, cur_step) * warmup_steps**(-1.5))

        return LambdaLR(optimizer, lr_lambda)

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
