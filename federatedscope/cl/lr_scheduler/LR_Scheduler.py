import numpy as np
from federatedscope.register import register_scheduler


# LR Scheduler
class LR_Scheduler(object):
    def __init__(self,
                 optimizer,
                 warmup_epochs,
                 warmup_lr,
                 num_epochs,
                 base_lr,
                 final_lr,
                 iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group[
                    'name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def get_scheduler(optimizer, type):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if type == 'cos_lr_scheduler':
        if optim is not None:
            lr_lambda = [lambda epoch: epoch // 30]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    warmup_epochs=0,
                                                    warmup_lr=0,
                                                    num_epochs=50,
                                                    base_lr=30,
                                                    final_lr=0,
                                                    iter_per_epoch=int(50000 /
                                                                       512))
        return scheduler


register_scheduler('cos_lr_scheduler', get_scheduler)
