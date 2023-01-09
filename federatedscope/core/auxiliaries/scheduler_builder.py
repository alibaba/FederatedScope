import copy
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
    # in case of users have not called the cfg.freeze()
    tmp_kwargs = copy.deepcopy(kwargs)
    if '__help_info__' in tmp_kwargs:
        del tmp_kwargs['__help_info__']
    if '__cfg_check_funcs__' in tmp_kwargs:
        del tmp_kwargs['__cfg_check_funcs__']
    if 'is_ready_for_run' in tmp_kwargs:
        del tmp_kwargs['is_ready_for_run']
    if 'warmup_ratio' in tmp_kwargs:
        del tmp_kwargs['warmup_ratio']
    if 'warmup_steps' in tmp_kwargs:
        warmup_steps = tmp_kwargs['warmup_steps']
        del tmp_kwargs['warmup_steps']
    if 'total_steps' in tmp_kwargs:
        total_steps = tmp_kwargs['total_steps']
        del tmp_kwargs['total_steps']

    for func in register.scheduler_dict.values():
        scheduler = func(optimizer, type, **tmp_kwargs)
        if scheduler is not None:
            return scheduler

    if type == 'warmup_step':
        from torch.optim.lr_scheduler import LambdaLR

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

        def lr_lambda(cur_step):
            return min(
                max(1, cur_step)**(-0.5),
                max(1, cur_step) * warmup_steps**(-1.5))

        return LambdaLR(optimizer, lr_lambda)

    if torch is None or type == '':
        return None
    if isinstance(type, str):
        if hasattr(torch.optim.lr_scheduler, type):
            return getattr(torch.optim.lr_scheduler, type)(optimizer,
                                                           **tmp_kwargs)
        else:
            raise NotImplementedError(
                'Scheduler {} not implement'.format(type))
    else:
        raise TypeError()
