from federatedscope.register import register_scheduler
from torch.optim.lr_scheduler import LambdaLR


def call_step_scheduler(optimizer,
                        type,
                        total_steps=-1,
                        warmup_steps=0,
                        **kwargs):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if type == 'step':

        def lr_lambda(cur_step):
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - cur_step) /
                float(max(1, total_steps - warmup_steps)))

        if optim is not None:
            scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler


def call_noam_scheduler(optimizer, type, warmup_steps=0, **kwargs):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if type == 'noam':

        def lr_lambda(cur_step):
            return min(
                max(1, cur_step)**(-0.5),
                max(1, cur_step) * warmup_steps**(-1.5))

        if optim is not None:
            scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler


register_scheduler('step', call_step_scheduler)
register_scheduler('noam', call_noam_scheduler)
