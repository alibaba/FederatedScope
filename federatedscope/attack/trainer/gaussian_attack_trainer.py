import logging
from typing import Type

import torch

from federatedscope.core.trainers import GeneralTorchTrainer

logger = logging.getLogger(__name__)


def wrap_GaussianAttackTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    wrap the gaussian attack trainer

    Args:
        base_trainer: Type: core.trainers.GeneralTorchTrainer
    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer
    '''

    base_trainer.replace_hook_in_train(
        new_hook=hook_on_batch_backward_generate_gaussian_noise_gradient,
        target_trigger='on_batch_backward',
        target_hook_name='_hook_on_batch_backward')

    return base_trainer


def hook_on_batch_backward_generate_gaussian_noise_gradient(ctx):
    ctx.optimizer.zero_grad()
    ctx.loss_task.backward()

    grad_values = list()
    for name, param in ctx.model.named_parameters():
        if 'bn' not in name:
            grad_values.append(param.grad.detach().cpu().view(-1))

    grad_values = torch.cat(grad_values)
    mean_for_gaussian_noise = torch.mean(grad_values) + 0.1
    std_for_gaussian_noise = torch.std(grad_values)

    for name, param in ctx.model.named_parameters():
        if 'bn' not in name:
            generated_grad = torch.normal(mean=mean_for_gaussian_noise,
                                          std=std_for_gaussian_noise,
                                          size=param.grad.shape)
            param.grad = generated_grad.to(param.grad.device)

    ctx.optimizer.step()
