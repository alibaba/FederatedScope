import copy
import torch
import logging

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from typing import Type

logger = logging.getLogger(__name__)


def wrap_FedRepTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    # ---------------------------------------------------------------------- #
    # FedRep method:
    # https://arxiv.org/abs/2102.07078
    # First training linear classifier and then feature extractor
    # Linear classifier: local_param; feature extractor: global_param
    # ---------------------------------------------------------------------- #
    init_FedRep_ctx(base_trainer)

    base_trainer.register_hook_in_train(new_hook=hook_on_fit_start_fedrep,
                                        trigger="on_fit_start",
                                        insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=hook_on_epoch_start_fedrep,
                                        trigger="on_epoch_start",
                                        insert_pos=-1)

    return base_trainer


def init_FedRep_ctx(base_trainer):

    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.epoch_feature = cfg.personalization.epoch_feature
    ctx.epoch_linear = cfg.personalization.epoch_linear

    ctx.num_train_epoch = ctx.epoch_feature + ctx.epoch_linear

    ctx.epoch_number = 0

    ctx.lr_feature = cfg.personalization.lr_feature
    ctx.lr_linear = cfg.personalization.lr_linear
    ctx.weight_decay = cfg.personalization.weight_decay

    ctx.local_param = cfg.personalization.local_param

    ctx.local_update_param = []
    ctx.global_update_param = []

    for name, param in ctx.model.named_parameters():
        if name.split(".")[0] in ctx.local_param:
            ctx.local_update_param.append(param)
        else:
            ctx.global_update_param.append(param)


def hook_on_fit_start_fedrep(ctx):

    ctx.num_train_epoch = ctx.epoch_feature + ctx.epoch_linear
    ctx.epoch_number = 0

    ctx.optimizer_for_feature = torch.optim.SGD(ctx.global_update_param,
                                                lr=ctx.lr_feature,
                                                momentum=0,
                                                weight_decay=ctx.weight_decay)
    ctx.optimizer_for_linear = torch.optim.SGD(ctx.local_update_param,
                                               lr=ctx.lr_linear,
                                               momentum=0,
                                               weight_decay=ctx.weight_decay)

    for name, param in ctx.model.named_parameters():

        if name.split(".")[0] in ctx.local_param:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ctx.optimizer = ctx.optimizer_for_linear


def hook_on_epoch_start_fedrep(ctx):

    ctx.epoch_number += 1

    if ctx.epoch_number == ctx.epoch_linear + 1:

        for name, param in ctx.model.named_parameters():

            if name.split(".")[0] in ctx.local_param:
                param.requires_grad = False
            else:
                param.requires_grad = True

        ctx.optimizer = ctx.optimizer_for_feature
