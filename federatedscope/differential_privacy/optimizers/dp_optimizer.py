from federatedscope.register import register_optimizer

import torch
import numpy as np
import abc

import logging

logger = logging.getLogger(__name__)


class DPMechanism(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def inject(self, shape):
        pass


class LaplaceMechanism(DPMechanism):
    """Laplace differential privacy mechanism that satisfies (\epsilon, 0)-differential privacy.

    Args:
         l1_norm_sensitivity (float): the l1-norm sensitivity of the data
         epsilon (float): the privacy bound for differential privacy

    Note:
        The injected laplace noise follows laplace distribution with argument `b`:

        \begin{equation}
            Laplace(x|b) = 1 / 2b * exp(-|x| / b)
        \end{equation}

        where `b` is calculated as `l1_norm_sensitivity / epsilon`, and the scale of the noise is `\sqrt{2} * b`

    """
    def __init__(self, l1_norm_sensitivity, epsilon, *args, **kwargs):
        self.scale = 2 ** 0.5 * l1_norm_sensitivity / epsilon

    def inject(self, param):
        if isinstance(param, torch.Tensor):
            size = param.size()
            laplace_distribution = torch.distributions.laplace.Laplace(loc=0., scale=self.scale)
            noise = laplace_distribution.sample(sample_shape=size).to(device=param.device)
            param.data += noise
        elif isinstance(param, np.ndarray):
            size = param.shape
            noise = np.random.laplace(loc=0, scale=self.scale, size=size)
            param += noise
        else:
            raise TypeError(f"Noise injector not support {type(param)}, expect torch.Tensor or np.ndarray.")


class GaussianMechanism(DPMechanism):
    """Gaussian differential privacy mechanism

    Args:
        l2_norm_sensitivity: l2 norm sensitivity
        noise_multiplier: used to calculate the std, e.g., scale = l2_norm_sensitivity * noise_multiplier
    """

    def __init__(self, l2_norm_sensitivity, noise_multiplier, *args, **kwargs):
        self.scale = l2_norm_sensitivity * noise_multiplier

    def inject(self, param):
        if isinstance(param, torch.Tensor):
            size = param.size()
            noise = torch.normal(mean=0, std=self.scale, size=size, device=param.device)
            param.data += noise
        elif isinstance(param, np.ndarry):
            size = param.shape
            noise = np.random.normal(loc=0, scale=self.scale, size=size)
            param += noise
        else:
            raise TypeError(f"Noise injector not support {type(param)}, expect torch.Tensor or np.ndarray.")


def wrap_dp_optimizer(cls, dp_mechanism):
    """Wrap a specific type of optimizer with specific differential privacy mechanism

    Args:
        cls: the optimizer class
        dp_mechanism: the class for specific differential privacy mechanism

    Returns:
        wrapped optimizer class
    """
    class DPOptimizer(cls, dp_mechanism):
        def __init__(self,
                     *args,
                     l1_norm_sensitivity=0.,
                     l2_norm_sensitivity=0.,
                     epsilon=1.,
                     noise_multiplier=1.,
                     **kwargs):
            cls.__init__(self, *args, **kwargs)
            dp_mechanism.__init__(self,
                                  epsilon=epsilon,
                                  noise_multiplier=noise_multiplier,
                                  l1_norm_sensitivity=l1_norm_sensitivity,
                                  l2_norm_sensitivity=l2_norm_sensitivity)

            logger.info(f'The DP optimizer will inject {dp_mechanism.__name__} noise with mean 0 and scale {self.scale}.')

        def step(self, *args, **kwargs):
            # Inject noise into the gradient
            for group in self.param_groups:
                params = group['params']
                for param in params:
                    if param.grad is not None:
                        self.inject(param)

            super(DPOptimizer, self).step(*args, **kwargs)

    return DPOptimizer


DPGaussianSGD = wrap_dp_optimizer(torch.optim.SGD, GaussianMechanism)

DPLaplaceSGD = wrap_dp_optimizer(torch.optim.SGD, LaplaceMechanism)


def call_dp_optimizer(type):
    if type.lower() == 'dpgaussiansgd':
        return DPGaussianSGD
    elif type.lower() == 'dplaplacesgd':
        return DPLaplaceSGD


register_optimizer('DPGaussianSGD', call_dp_optimizer)
