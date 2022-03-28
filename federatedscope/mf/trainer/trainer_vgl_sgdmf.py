from federatedscope.mf.trainer.trainer import MFTrainer
from federatedscope.core.auxiliaries.utils import get_random
from typing import Type
import numpy as np

import torch


def wrap_VFLTrainer(base_trainer: Type[MFTrainer]) -> Type[MFTrainer]:
    """Build a `VFLTrainer` with a plug-in manner, by registering new functions into specific `MFTrainer`

    """

    # ---------------- attribute-level plug-in -----------------------
    init_sgdmf_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    del_one_hook_idx = base_trainer.reset_hook_in_train(
        target_trigger="on_batch_backward",
        target_hook_name="_hook_on_batch_backward")
    base_trainer.register_hook_in_train(new_hook=hook_on_batch_backward,
                                        trigger="on_batch_backward",
                                        insert_pos=del_one_hook_idx)
    return base_trainer


def init_sgdmf_ctx(base_trainer):
    """Init necessary attributes used in SGDMF,
    some new attributes will be with prefix `SGDMF` optimizer to avoid namespace pollution
    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    sample_ratio = float(cfg.data.batch_size) / cfg.model.num_user
    # Noise multiplier
    tmp = cfg.sgdmf.constant * np.power(sample_ratio, 2) * (
        cfg.federate.total_round_num * ctx.num_total_train_batch +
        cfg.sgdmf.local_finetune_steps) * np.log(1. / cfg.sgdmf.delta)
    ctx.sgdmf_z = np.sqrt(tmp / np.power(cfg.sgdmf.epsilon, 2))

    ctx.sgdmf_R = cfg.sgdmf.R
    ctx.sgdmf_theta = cfg.sgdmf.theta


def hook_on_batch_backward(ctx):
    """Private local updates in SGDMF

    """
    ctx.optimizer.zero_grad()
    ctx.loss_task.backward()
    # Insert noise
    scale_user = float(ctx.sgdmf_theta) * ctx.sgdmf_z * np.power(
        ctx.sgdmf_R, 1.5)
    ctx.model.embed_user.grad.data += get_random(
        "Normal",
        sample_shape=ctx.model.embed_user.shape,
        params={
            "loc": 0,
            "scale": scale_user
        },
        device=ctx.model.embed_user.device)
    scale_item = ctx.sgdmf_z * np.power(ctx.sgdmf_R, 1.5)
    ctx.model.embed_item.grad.data += get_random(
        "Normal",
        sample_shape=ctx.model.embed_item.shape,
        params={
            "loc": 0,
            "scale": scale_item
        },
        device=ctx.model.embed_item.device)
    ctx.optimizer.step()
    # Embedding clipping
    for vec in ctx.model.embed_user:
        vec.data = vec.data / max(
            1.,
            torch.norm(vec, p=2).item() / np.sqrt(ctx.sgdmf_R))
