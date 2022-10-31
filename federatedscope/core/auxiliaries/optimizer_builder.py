import copy
import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    from federatedscope.contrib.optimizer import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.optimizer`, some modules are not '
        f'available.')


def get_optimizer(model, opt_type, lr, **kwargs):
    """
    This function returns an instantiated optimizer to optimize the model.

    Args:
        model: model to be optimized
        opt_type: type of optimizer, see \
          https://pytorch.org/docs/stable/optim.html
        lr: learning rate
        **kwargs: kwargs dict

    Returns:
        An instantiated optimizer
    """
    if torch is None:
        return None
    # in case of users have not called the cfg.freeze()
    tmp_kwargs = copy.deepcopy(kwargs)
    if '__help_info__' in tmp_kwargs:
        del tmp_kwargs['__help_info__']
    if '__cfg_check_funcs__' in tmp_kwargs:
        del tmp_kwargs['__cfg_check_funcs__']
    if 'is_ready_for_run' in tmp_kwargs:
        del tmp_kwargs['is_ready_for_run']

    for func in register.optimizer_dict.values():
        optimizer = func(model, opt_type, lr, **tmp_kwargs)
        if optimizer is not None:
            return optimizer

    if isinstance(opt_type, str):
        if hasattr(torch.optim, opt_type):
            if isinstance(model, torch.nn.Module):
                return getattr(torch.optim, opt_type)(model.parameters(), lr,
                                                      **tmp_kwargs)
            else:
                return getattr(torch.optim, opt_type)(model, lr, **tmp_kwargs)
        else:
            raise NotImplementedError(
                'Optimizer {} not implement'.format(opt_type))
    else:
        raise TypeError()
