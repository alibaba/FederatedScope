from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(type, optimizer, total_steps, warmup_steps=0, last_epoch=-1, **kwargs):
    if isinstance(type, str):
        if type == 'step':
            def lr_lambda(cur_step):
                if cur_step < warmup_steps:
                    return float(cur_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - cur_step) / float(max(1, total_steps - warmup_steps)))

            return LambdaLR(optimizer, lr_lambda, last_epoch)
        else:
            raise NotImplementedError('Scheduler {} not implement'.format(type))
    else:
        raise TypeError()
