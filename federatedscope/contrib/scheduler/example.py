from federatedscope.register import register_scheduler


def call_my_scheduler(optimizer, type):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if type == 'myscheduler':
        if optim is not None:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer)
        return scheduler


register_scheduler('myscheduler', call_my_scheduler)
