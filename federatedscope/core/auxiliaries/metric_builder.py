import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.metrics import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.metrics`, some modules are not '
        f'available.')


def get_metric(types):
    metrics = dict()
    for func in register.metric_dict.values():
        res = func(types)
        if res is not None:
            name, metric = res
            metrics[name] = metric
    return metrics
