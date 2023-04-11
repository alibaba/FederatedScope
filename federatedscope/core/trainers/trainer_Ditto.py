import copy
import logging

import torch

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
# from federatedscope.core.auxiliaries.utils import calculate_batch_epoch_num
from typing import Type

logger = logging.getLogger(__name__)

DEBUG_DITTO = False


def wrap_DittoTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """
    Build a `DittoTrainer` with a plug-in manner, by registering new
    functions into specific `BaseTrainer`

    The Ditto implementation, "Ditto: Fair and Robust Federated Learning
    Through Personalization. (ICML2021)"
    based on the Algorithm 2 in their paper and official codes:
    https://github.com/litian96/ditto
    """

    # ---------------- attribute-level plug-in -----------------------
    init_Ditto_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.register_hook_in_train(new_hook=_hook_on_fit_start_clean,
                                        trigger='on_fit_start',
                                        insert_pos=-1)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_fit_start_set_regularized_para,
        trigger="on_fit_start",
        insert_pos=0)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_start_switch_model,
        trigger="on_batch_start",
        insert_pos=0)
    base_trainer.register_hook_in_train(new_hook=hook_on_batch_forward_cnt_num,
                                        trigger="on_batch_forward",
                                        insert_pos=-1)
    base_trainer.register_hook_in_train(new_hook=_hook_on_batch_end_flop_count,
                                        trigger="on_batch_end",
                                        insert_pos=-1)
    base_trainer.register_hook_in_eval(
        new_hook=hook_on_fit_start_switch_local_model,
        trigger="on_fit_start",
        insert_pos=0)
    base_trainer.register_hook_in_eval(
        new_hook=hook_on_fit_end_switch_global_model,
        trigger="on_fit_end",
        insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=hook_on_fit_end_free_cuda,
                                        trigger="on_fit_end",
                                        insert_pos=-1)
    base_trainer.register_hook_in_eval(new_hook=hook_on_fit_end_free_cuda,
                                       trigger="on_fit_end",
                                       insert_pos=-1)

    return base_trainer


def init_Ditto_ctx(base_trainer):
    """
    init necessary attributes used in Ditto,
    `global_model` acts as the shared global model in FedAvg;
    `local_model` acts as personalized model will be optimized with
    regularization based on weights of `global_model`

    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.global_model = copy.deepcopy(ctx.model)
    ctx.local_model = copy.deepcopy(ctx.model)
    ctx.models = [ctx.local_model, ctx.global_model]

    ctx.model = ctx.global_model
    ctx.use_local_model_current = False

    ctx.num_samples_local_model_train = 0

    ctx.num_train_batch_for_local_model, \
        ctx.num_train_batch_last_epoch_for_local_model, \
        ctx.num_train_epoch_for_local_model, \
        ctx.num_total_train_batch \
        = ctx.pre_calculate_batch_epoch_num \
        (cfg.personalization.local_update_steps)

    if cfg.federate.batch_or_epoch == 'batch':
        ctx.num_train_batch += ctx.num_train_batch_for_local_model
        ctx.num_train_batch_last_epoch += \
            ctx.num_train_batch_last_epoch_for_local_model
    else:
        ctx.num_train_epoch += ctx.num_train_epoch_for_local_model


def hook_on_fit_start_set_regularized_para(ctx):
    # set the compared model data for local personalized model
    ctx.global_model.to(ctx.device)
    ctx.local_model.to(ctx.device)
    ctx.global_model.train()
    ctx.local_model.train()
    compared_global_model_para = [{
        "params": list(ctx.global_model.parameters())
    }]

    ctx.optimizer_for_global_model = get_optimizer(ctx.global_model,
                                                   **ctx.cfg.optimizer)
    ctx.optimizer_for_local_model = get_optimizer(ctx.local_model,
                                                  **ctx.cfg.optimizer)

    ctx.optimizer_for_local_model = \
        wrap_regularized_optimizer(ctx.optimizer_for_local_model, \
                                   ctx.cfg.personalization.regular_weight)

    ctx.optimizer_for_local_model.set_compared_para_group(
        compared_global_model_para)


def _hook_on_fit_start_clean(ctx):
    # remove the unnecessary optimizer
    del ctx.optimizer
    ctx.num_samples_local_model_train = 0


def _hook_on_batch_end_flop_count(ctx):

    ctx.monitor.total_flops += ctx.monitor.total_model_size / 2


def hook_on_batch_forward_cnt_num(ctx):
    if ctx.use_local_model_current:
        ctx.num_samples_local_model_train += ctx.batch_size


def hook_on_batch_start_switch_model(ctx):
    if ctx.cfg.federate.batch_or_epoch == 'batch':
        if ctx.cur_epoch_i == (ctx.num_train_epoch - 1):
            ctx.use_local_model_current = \
                ctx.cur_batch_i < \
                ctx.num_train_batch_last_epoch_for_local_model
        else:
            ctx.use_local_model_current = \
                ctx.cur_batch_i < ctx.num_train_batch_for_local_model
    else:
        ctx.use_local_model_current = \
            ctx.cur_epoch_i < ctx.num_train_epoch_for_local_model

    if DEBUG_DITTO:
        logger.info("====================================================")
        logger.info(f"cur_epoch_i: {ctx.cur_epoch_i}")
        logger.info(f"num_train_epoch: {ctx.num_train_epoch}")
        logger.info(f"cur_batch_i: {ctx.cur_batch_i}")
        logger.info(f"num_train_batch: {ctx.num_train_batch}")
        logger.info(f"num_train_batch_for_local_model: "
                    f"{ctx.num_train_batch_for_local_model}")
        logger.info(f"num_train_epoch_for_local_model: "
                    f"{ctx.num_train_epoch_for_local_model}")
        logger.info(f"use_local_model: {ctx.use_local_model_current}")

    if ctx.use_local_model_current:
        ctx.model = ctx.local_model
        ctx.optimizer = ctx.optimizer_for_local_model
    else:
        ctx.model = ctx.global_model
        ctx.optimizer = ctx.optimizer_for_global_model


def hook_on_fit_start_switch_local_model(ctx):
    ctx.model = ctx.local_model
    ctx.model.eval()


def hook_on_fit_end_switch_global_model(ctx):
    ctx.model = ctx.global_model


def hook_on_fit_end_free_cuda(ctx):
    ctx.global_model.to(torch.device("cpu"))
    ctx.local_model.to(torch.device("cpu"))
