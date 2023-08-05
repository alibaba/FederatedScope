import logging

from federatedscope.mf.trainer.trainer import MFTrainer
from federatedscope.core.auxiliaries.utils import get_random
from typing import Type
import numpy as np

import torch

logger = logging.getLogger(__name__)


def wrap_MFTrainer(base_trainer: Type[MFTrainer]) -> Type[MFTrainer]:
    """Build `SGDMFTrainer` with a plug-in manner, by registering new
    functions into specific `MFTrainer`

    """

    # ---------------- attribute-level plug-in -----------------------
    init_sgdmf_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.replace_hook_in_train(
        new_hook=hook_on_batch_backward,
        target_trigger="on_batch_backward",
        target_hook_name="_hook_on_batch_backward")

    return base_trainer


def init_sgdmf_ctx(base_trainer):
    """Init necessary attributes used in SGDMF,
    some new attributes will be with prefix `SGDMF` optimizer to avoid
    namespace pollution

    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    sample_ratio = float(cfg.data.batch_size) / cfg.model.num_user
    # Noise multiplier
    tmp = cfg.sgdmf.constant * np.power(sample_ratio, 2) * (
        cfg.federate.total_round_num * ctx.num_total_train_batch) * np.log(
            1. / cfg.sgdmf.delta)
    noise_multipler = np.sqrt(tmp / np.power(cfg.sgdmf.epsilon, 2))
    ctx.scale = max(cfg.sgdmf.theta, 1.) * noise_multipler * np.power(
        cfg.sgdmf.R, 1.5)
    logger.info("Inject noise: (loc=0, scale={})".format(ctx.scale))
    ctx.sgdmf_R = cfg.sgdmf.R


def embedding_clip(param, R: int):
    """Clip embedding vector according to $R$

    Arguments:
        param (tensor): The embedding vector
        R (int): The upper bound of ratings
    """
    # Turn all negative entries of U into 0
    param.data = (torch.abs(param.data) + param.data) * 0.5
    # Clip tensor
    norms = torch.linalg.norm(param.data, dim=1)
    threshold = np.sqrt(R)
    param.data[norms > threshold] *= (threshold /
                                      norms[norms > threshold]).reshape(
                                          (-1, 1))
    param.data[param.data < 0] = 0.


def hook_on_batch_backward(ctx):
    """Private local updates in SGDMF

    """
    ctx.optimizer.zero_grad()
    ctx.loss_task.backward()

    # Inject noise
    ctx.model.embed_user.grad.data += get_random(
        "Normal",
        sample_shape=ctx.model.embed_user.shape,
        params={
            "loc": 0,
            "scale": ctx.scale
        },
        device=ctx.model.embed_user.device)
    ctx.model.embed_item.grad.data += get_random(
        "Normal",
        sample_shape=ctx.model.embed_item.shape,
        params={
            "loc": 0,
            "scale": ctx.scale
        },
        device=ctx.model.embed_item.device)
    ctx.optimizer.step()

    # Embedding clipping
    with torch.no_grad():
        embedding_clip(ctx.model.embed_user, ctx.sgdmf_R)
        embedding_clip(ctx.model.embed_item, ctx.sgdmf_R)
