from federatedscope.register import register_optimizer


def call_my_optimizer(type):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        optimizer = None

    if type.lower() == 'myoptimizer':
        if optim is not None:
            optimizer = optim.Adam
        return optimizer


register_optimizer('myoptimizer', call_my_optimizer)
