import copy
import torch
import logging
import math

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer

from typing import Type

logger = logging.getLogger(__name__)


def wrap_Simple_tuning_Trainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    # ---------------------------------------------------------------------- #
    # Simple_tuning method:
    # https://arxiv.org/abs/2302.01677
    # Only tuning the linear classifier and freeze the feature extractor
    # the key is to reinitialize the linear classifier
    # ---------------------------------------------------------------------- #
    init_Simple_tuning_ctx(base_trainer)

    base_trainer.register_hook_in_ft(new_hook=hook_on_fit_start_simple_tuning,
                                     trigger="on_fit_start",
                                     insert_pos=-1)

    return base_trainer


def init_Simple_tuning_ctx(base_trainer):

    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.epoch_linear = cfg.finetune.epoch_linear

    ctx.num_train_epoch = ctx.epoch_linear

    ctx.epoch_number = 0

    ctx.lr_linear = cfg.finetune.lr_linear
    ctx.weight_decay = cfg.finetune.weight_decay

    ctx.local_param = cfg.finetune.local_param

    ctx.local_update_param = []

    for name, param in ctx.model.named_parameters():
        if name.split(".")[0] in ctx.local_param:
            ctx.local_update_param.append(param)


def hook_on_fit_start_simple_tuning(ctx):

    ctx.num_train_epoch = ctx.epoch_linear
    ctx.epoch_number = 0

    ctx.optimizer_for_linear = torch.optim.SGD(ctx.local_update_param,
                                               lr=ctx.lr_linear,
                                               momentum=0,
                                               weight_decay=ctx.weight_decay)

    for name, param in ctx.model.named_parameters():
        if name.split(".")[0] in ctx.local_param:
            if name.split(".")[1] == 'weight':
                stdv = 1. / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)
            else:
                param.data.uniform_(-stdv, stdv)
            param.requires_grad = True
        else:
            param.requires_grad = False

    ctx.optimizer = ctx.optimizer_for_linear
