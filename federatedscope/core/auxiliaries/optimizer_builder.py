try:
    import torch
except ImportError:
    torch = None

import copy


def get_optimizer(model, type, lr, **kwargs):
    if torch is None:
        return None
    # in case of users have not called the cfg.freeze()
    tmp_kwargs = copy.deepcopy(kwargs)
    if '__help_info__' in tmp_kwargs:
        del tmp_kwargs['__help_info__']
    if '__cfg_check_funcs__' in tmp_kwargs:
        del tmp_kwargs['__cfg_check_funcs__']
    if 'is_ready_for_run' in tmp_kwargs:
        del tmp_kwargs['is_ready_for_run']
    if isinstance(type, str):
        if hasattr(torch.optim, type):
            if isinstance(model, torch.nn.Module):
                return getattr(torch.optim, type)(model.parameters(), lr,
                                                  **tmp_kwargs)
            else:
                return getattr(torch.optim, type)(model, lr, **tmp_kwargs)
        else:
            raise NotImplementedError(
                'Optimizer {} not implement'.format(type))
    else:
        raise TypeError()

    return wrap_optimizer(optimizer)


class Injector(object):
    def __init__(self, sensitivity, noise_multiplier):
        pass

    def generate(self, shape):
        pass

class LaplaceInjector(Injector):
    def __init__(self, sensitivity, noise_multiplier):
        self.scale = 1

    def inject(self, device):
        pass


import numpy as np

class GaussianInjector(Injector):
    def __init__(self, sensitivity, noise_multiplier):
        self.scale = sensitivity * noise_multiplier

    def inject(self, param):
        if isinstance(param, torch.Tensor):
            size = param.size()
            noise = torch.normal(mean=0,  std=self.scale, size=size, device=param.device)
            param.data += noise
        elif isinstance(param, np.ndarry):
            size = param.shape
            noise = np.random.normal(loc=0, scale=self.scale, size=size)
            param += noise
        else:
            raise TypeError(f"Noise injector not support {type(param)}, expect torch.Tensor or np.ndarray.")




class DPOptimizer(torch.optim.SGD):
    def __init__(self, injector, *args, **kwargs):
        super(DPOptimizer, self).__init__(*args, **kwargs)
        self.injector = injector

    def step(self, *args, **kwargs):
        # Clip gradient and inject noise
        torch.nn.utils.clip_grad_norm()

        for group in self.param_groups:
            params = group['params']
            for param in params:
                if param.grad is not None:

                    self.injector.inject(param)

        super(DPOptimizer, self).step()


if __name__ == '__main__':
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x):
            return self.linear(x)


    net = Net()
    opt = DPOptimizer(GaussianInjector(1, 0.1), net.parameters(), lr=0.01)
    opt.step()
