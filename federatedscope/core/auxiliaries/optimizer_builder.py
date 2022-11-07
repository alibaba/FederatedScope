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
    from federatedscope.differential_privacy.optimizers import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.optimizer` or `federatedscope.differential_privacy.optimizers`, some modules are not '
        f'available.')


def get_optimizer(model, type, lr, **kwargs):
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
    if isinstance(type, str):
        if hasattr(torch.optim, type):
            if isinstance(model, torch.nn.Module):
                return getattr(torch.optim, type)(model.parameters(), lr,
                                                  **tmp_kwargs)
            else:
                return getattr(torch.optim, type)(model, lr, **tmp_kwargs)
        else:
            # registered optimizers
            for func in register.optimizer_dict.values():
                optimizer = func(type)
                if optimizer is not None:
                    return optimizer(model.parameters(), lr, **tmp_kwargs)

            raise NotImplementedError(
                'Optimizer {} not implement'.format(type))
    else:
        raise TypeError()
