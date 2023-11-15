import logging
import federatedscope.register as register
from federatedscope.nlp.hetero_tasks.metric import *

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.metrics import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.metrics`, some modules are not '
        f'available.')


def get_metric(types):
    """
    This function returns a dict, where the key is metric name, and value is \
    the function of how to calculate the metric and a bool to indicate the \
    metric is larger the better.

    Args:
        types: list of metric names

    Returns:
        A metric calculator dict, such as \
        ``{'loss': (eval_loss, False), 'acc': (eval_acc, True), ...}``

    Note:
      The key-value pairs of built-in metric and related funcs and \
      ``the_larger_the_better`` sign is shown below:
        =================  =============================================  =====
        Metric name        Source                                         \
        The larger the better
        =================  =============================================  =====
        ``loss``           ``monitors.metric_calculator.eval_loss``       False
        ``avg_loss``       ``monitors.metric_calculator.eval_avg_loss``   False
        ``total``          ``monitors.metric_calculator.eval_total``      False
        ``correct``        ``monitors.metric_calculator.eval_correct``    True
        ``acc``            ``monitors.metric_calculator.eval_acc``        True
        ``ap``             ``monitors.metric_calculator.eval_ap``         True
        ``f1``             ``monitors.metric_calculator.eval_f1_score``   True
        ``roc_auc``        ``monitors.metric_calculator.eval_roc_auc``    True
        ``rmse``           ``monitors.metric_calculator.eval_rmse``       False
        ``mse``            ``monitors.metric_calculator.eval_mse``        False
        ``loss_regular``   ``monitors.metric_calculator.eval_regular``    False
        ``imp_ratio``      ``monitors.metric_calculator.eval_imp_ratio``  True
        ``std``            ``None``                                       False
        ``hits@{n}``       ``monitors.metric_calculator.eval_hits``       True
        =================  =============================================  =====
    """
    metrics = dict()
    for func in register.metric_dict.values():
        res = func(types)
        if res is not None:
            name, metric, the_larger_the_better = res
            metrics[name] = metric, the_larger_the_better
    for key in types:
        if key not in metrics.keys():
            logger.warning(f'eval.metrics `{key}` method not found!')
    return metrics
