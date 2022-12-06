from federatedscope.register import register_scheduler


def call_my_scheduler(optimizer, reg_type, **kwargs):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if reg_type == 'myscheduler':
        if optim is not None:
            lr_lambda = [lambda epoch: epoch // 30]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler


register_scheduler('myscheduler', call_my_scheduler)
