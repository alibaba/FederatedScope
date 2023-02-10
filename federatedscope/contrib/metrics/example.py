from federatedscope.register import register_metric

METRIC_NAME = 'example'


def MyMetric(ctx, **kwargs):
    return ctx.num_train_data


def call_my_metric(types):
    if METRIC_NAME in types:
        the_larger_the_better = True
        metric_builder = MyMetric
        return METRIC_NAME, metric_builder, the_larger_the_better


register_metric(METRIC_NAME, call_my_metric)
