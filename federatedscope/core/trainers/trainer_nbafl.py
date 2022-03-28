import logging

from federatedscope.core.auxiliaries.utils import get_random
from federatedscope.core.trainers.trainer import GeneralTrainer
from federatedscope.core.worker.server import Server
from federatedscope.core.message import Message
from typing import Type
from copy import deepcopy

import numpy as np
import torch


def wrap_nbafl_trainer(
        base_trainer: Type[GeneralTrainer]) -> Type[GeneralTrainer]:
    """Implementation of NbAFL refer to `Federated Learning with Differential Privacy: Algorithms and Performance Analysis` [et al., 2020]
        (https://ieeexplore.ieee.org/abstract/document/9069945/)

        Arguments:
            mu: the factor of the regularizer
            epsilon: the distinguishable bound
            w_clip: the threshold to clip weights

    """

    # ---------------- attribute-level plug-in -----------------------
    init_nbafl_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.register_hook_in_train(new_hook=record_initialization,
                                        trigger='on_fit_start',
                                        insert_pos=-1)

    base_trainer.register_hook_in_eval(new_hook=record_initialization,
                                       trigger='on_fit_start',
                                       insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=del_initialization,
                                        trigger='on_fit_end',
                                        insert_pos=-1)

    base_trainer.register_hook_in_eval(new_hook=del_initialization,
                                       trigger='on_fit_end',
                                       insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=inject_noise_in_upload,
                                        trigger='on_fit_end',
                                        insert_pos=-1)
    return base_trainer


def init_nbafl_ctx(base_trainer):
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    cfg.regularizer.type = 'proximal_regularizer'
    cfg.regularizer.mu = cfg.nbafl.mu

    from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
    ctx.regularizer = get_regularizer(cfg.regularizer.type)

    ctx.nbafl_epsilon = cfg.nbafl.epsilon
    ctx.nbafl_w_clip = cfg.nbafl.w_clip
    ctx.nbafl_constant = cfg.nbafl.constant

    ctx.nbafl_total_round_num = cfg.federate.total_round_num


# ------------------------------------------------------------------------ #
# Additional functions for NbAFL algorithm
# ------------------------------------------------------------------------ #


# Trainer
def record_initialization(ctx):
    ctx.weight_init = deepcopy(
        [_.data.detach() for _ in ctx.model.parameters()])


def del_initialization(ctx):
    ctx.weight_init = None


def inject_noise_in_upload(ctx):
    """Inject noise into weights before the client upload them to server

    """
    scale_u = ctx.nbafl_w_clip * ctx.nbafl_total_round_num * 2 * ctx.nbafl_constant / ctx.num_train_data / ctx.nbafl_epsilon
    # logging.info({"Role": "Client", "Noise": {"mean": 0, "scale": scale_u}})
    for p in ctx.model.parameters():
        noise = get_random("Normal", p.shape, {
            "loc": 0,
            "scale": scale_u
        }, p.device)
        p.data += noise


# Server
def inject_noise_in_broadcast(cfg, sample_client_num, model):
    """Inject noise into weights before the server broadcasts them

    """

    if len(sample_client_num) == 0:
        return

    # Clip weight
    for p in model.parameters():
        p.data = p.data / torch.max(torch.ones(size=p.shape),
                                    torch.abs(p.data) / cfg.nbafl.w_clip)

    if len(sample_client_num) > 0:
        # Inject noise
        L = cfg.federate.sample_client_num if cfg.federate.sample_client_num > 0 else cfg.federate.client_num
        if cfg.federate.total_round_num > np.sqrt(cfg.federate.client_num) * L:
            scale_d = 2 * cfg.nbafl.w_clip * cfg.nbafl.constant * np.sqrt(
                np.power(cfg.federate.total_round_num, 2) -
                np.power(L, 2) * cfg.federate.client_num) / (
                    min(sample_client_num.values()) * cfg.federate.client_num *
                    cfg.nbafl.epsilon)
            # logging.info({"Role": "Server", "Noise": {"mean": 0, "scale": scale_d}})
            for p in model.parameters():
                p.data += get_random("Normal", p.shape, {
                    "loc": 0,
                    "scale": scale_d
                }, p.device)


def wrap_nbafl_server(server: Type[Server]) -> Type[Server]:
    server.register_noise_injector(inject_noise_in_broadcast)
