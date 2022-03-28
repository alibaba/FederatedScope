from federatedscope.register import register_metric

METRIC_NAME = 'example'


def mymetric(ctx, **kwargs):
    return ctx["num_train_data"]


def call_example_metric(types):
    if METRIC_NAME in types:
        metric_builder = mymetric
        return METRIC_NAME, metric_builder


register_metric(METRIC_NAME, call_example_metric)
