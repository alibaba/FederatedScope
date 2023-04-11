from calendar import c
from email.mime import base
import logging
from typing import Type
import torch
import numpy as np
import copy

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.attack.auxiliary.backdoor_utils import normalize
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

# from federatedscope.attack.auxiliary.poisoning_dataset import load_poisoned_dataset_edgeset, load_poisoned_dataset_pixel

logger = logging.getLogger(__name__)


def wrap_benignTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    Warp the benign trainer for backdoor attack:
    
    We just add the normalization operation.

    Args:
        base_trainer: Type: core.trainers.GeneralTorchTrainer

    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer

    '''

    if base_trainer.cfg.attack.norm_clip:

        base_trainer.register_hook_in_train(
            new_hook=hook_on_fit_start_init_local_model,
            trigger='on_fit_start',
            insert_pos=-1)

        base_trainer.ctx.norm_clip_value = base_trainer.cfg.attack.norm_clip_value

        base_trainer.register_hook_in_train(
            new_hook=hook_on_fit_end_clip_model,
            trigger='on_fit_end',
            insert_pos=-1)

    if base_trainer.cfg.attack.dp_noise > 0.0:
        base_trainer.ctx.dp_noise = base_trainer.cfg.attack.dp_noise

    else:
        base_trainer.ctx.dp_noise = 0.0

    return base_trainer


def get_weight_difference(weight1, weight2):
    difference = {}
    res = []
    if type(weight2) == dict:
        for name, layer in weight1.items():
            difference[name] = layer.data - weight2[name].data
            res.append(difference[name].view(-1))
    else:
        for name, layer in weight2:
            difference[name] = weight1[name].data - layer.data
            res.append(difference[name].view(-1))

    difference_flat = torch.cat(res)

    return difference, difference_flat


def get_l2_norm(weight1, weight2):
    difference = {}
    res = []
    if type(weight2) == dict:
        for name, layer in weight1.items():
            difference[name] = layer.data - weight2[name].data
            res.append(difference[name].view(-1))
    else:
        for name, layer in weight2:
            difference[name] = weight1[name].data - layer.data
            res.append(difference[name].view(-1))

    difference_flat = torch.cat(res)

    l2_norm = torch.norm(difference_flat.clone().detach().cuda())

    l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())

    return l2_norm, l2_norm_np


def clip_grad(norm_bound, weight_difference, difference_flat):

    l2_norm = torch.norm(difference_flat.clone().detach().cuda())
    scale = max(1.0, float(torch.abs(l2_norm / norm_bound)))
    for name in weight_difference.keys():
        weight_difference[name].div_(scale)

    return weight_difference, l2_norm


def copy_params(model, target_params_variables):
    for name, layer in model.named_parameters():
        layer.data = copy.deepcopy(target_params_variables[name])


def hook_on_fit_start_init_local_model(ctx):

    ctx.global_model_copy = dict()
    for name, param in ctx.model.named_parameters():
        ctx.global_model_copy[name] = ctx.model.state_dict()[name].clone(
        ).detach().requires_grad_(False)


def hook_on_fit_end_clip_model(ctx):

    l2_norm, l2_norm_np = get_l2_norm(ctx.global_model_copy,
                                      ctx.model.named_parameters())
    logger.info('l2 norm of local model (before server defense):{}'.format(
        l2_norm.item()))
    weight_difference, difference_flat = get_weight_difference(
        ctx.global_model_copy, ctx.model.named_parameters())
    clipped_weight_difference, _ = clip_grad(ctx.norm_clip_value,
                                             weight_difference,
                                             difference_flat)

    for key_, para in clipped_weight_difference.items():
        clipped_weight_difference[
            key_] = para.data + ctx.dp_noise * torch.rand_like(
                copy.deepcopy(para.data))

    weight_difference, difference_flat = get_weight_difference(
        ctx.global_model_copy, clipped_weight_difference)
    copy_params(ctx.model, weight_difference)

    l2_norm, l2_norm_np = get_l2_norm(ctx.global_model_copy,
                                      ctx.model.named_parameters())
    logger.info('l2 norm of local model (after server defense):{}'.format(
        l2_norm.item()))


def hook_on_fit_end_test_poison(ctx):
    """Evaluate metrics of poisoning attacks.

    """

    ctx['poison_' + ctx.cur_data_split +
        '_loader'] = ctx.data['poison_' + ctx.cur_data_split]
    ctx['poison_' + ctx.cur_data_split +
        '_data'] = ctx.data['poison_' + ctx.cur_data_split].dataset
    ctx['num_poison_' + ctx.cur_data_split + '_data'] = len(
        ctx.data['poison_' + ctx.cur_data_split].dataset)
    setattr(ctx, "poison_{}_y_true".format(ctx.cur_data_split), [])
    setattr(ctx, "poison_{}_y_prob".format(ctx.cur_data_split), [])
    setattr(ctx, "poison_num_samples_{}".format(ctx.cur_data_split), 0)

    for batch_idx, (samples, targets) in enumerate(
            ctx['poison_' + ctx.cur_data_split + '_loader']):
        samples, targets = samples.to(ctx.device), targets.to(ctx.device)
        pred = ctx.model(samples)
        if len(targets.size()) == 0:
            targets = targets.unsqueeze(0)
        ctx.poison_y_true = targets
        ctx.poison_y_prob = pred
        ctx.poison_batch_size = len(targets)

        ctx.get("poison_{}_y_true".format(ctx.cur_data_split)).append(
            ctx.poison_y_true.detach().cpu().numpy())

        ctx.get("poison_{}_y_prob".format(ctx.cur_data_split)).append(
            ctx.poison_y_prob.detach().cpu().numpy())

        setattr(
            ctx, "poison_num_samples_{}".format(ctx.cur_data_split),
            ctx.get("poison_num_samples_{}".format(ctx.cur_data_split)) +
            ctx.poison_batch_size)

    setattr(
        ctx, "poison_{}_y_true".format(ctx.cur_data_split),
        np.concatenate(ctx.get("poison_{}_y_true".format(ctx.cur_data_split))))
    setattr(
        ctx, "poison_{}_y_prob".format(ctx.cur_data_split),
        np.concatenate(ctx.get("poison_{}_y_prob".format(ctx.cur_data_split))))

    logger.info('the {} poisoning samples: {:d}'.format(
        ctx.cur_data_split,
        ctx.get("poison_num_samples_{}".format(ctx.cur_data_split))))

    poison_true = ctx['poison_' + ctx.cur_data_split + '_y_true']
    poison_prob = ctx['poison_' + ctx.cur_data_split + '_y_prob']

    poison_pred = np.argmax(poison_prob, axis=1)

    correct = poison_true == poison_pred

    poisoning_acc = float(np.sum(correct)) / len(correct)

    logger.info('the {} poisoning accuracy: {:f}'.format(
        ctx.cur_data_split, poisoning_acc))
