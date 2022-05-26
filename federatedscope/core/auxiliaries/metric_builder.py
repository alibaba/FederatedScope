import federatedscope.register as register

try:
    from federatedscope.contrib.metrics import *
except ImportError:
    pass


def get_metric(types):
    metrics = dict()
    for func in register.metric_dict.values():
        res = func(types)
        if res is not None:
            name, metric = res
            metrics[name] = metric
    return metrics
