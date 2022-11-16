from federatedscope.core.trainers.utils import get_random
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from typing import Type
from copy import deepcopy

import numpy as np
import torch


def wrap_nbafl_trainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """Implementation of NbAFL refer to `Federated Learning with
    Differential Privacy: Algorithms and Performance Analysis` [et al., 2020]
        (https://ieeexplore.ieee.org/abstract/document/9069945/)

        Arguments:
            mu: the factor of the regularizer
            epsilon: the distinguishable bound
            w_clip: the threshold to clip weights

    """

    # ---------------- attribute-level plug-in -----------------------
    init_nbafl_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.register_hook_in_train(new_hook=_hook_record_initialization,
                                        trigger='on_fit_start',
                                        insert_pos=-1)

    base_trainer.register_hook_in_eval(new_hook=_hook_record_initialization,
                                       trigger='on_fit_start',
                                       insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=_hook_del_initialization,
                                        trigger='on_fit_end',
                                        insert_pos=-1)

    base_trainer.register_hook_in_eval(new_hook=_hook_del_initialization,
                                       trigger='on_fit_end',
                                       insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=_hook_inject_noise_in_upload,
                                        trigger='on_fit_end',
                                        insert_pos=-1)
    return base_trainer


def init_nbafl_ctx(base_trainer):
    """Set proximal regularizer, and the scale of gaussian noise

    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    # set proximal regularizer
    cfg.defrost()
    cfg.regularizer.type = 'proximal_regularizer'
    cfg.regularizer.mu = cfg.nbafl.mu
    cfg.freeze()
    from federatedscope.core.auxiliaries.regularizer_builder import \
        get_regularizer
    ctx.regularizer = get_regularizer(cfg.regularizer.type)

    # set noise scale during upload
    if cfg.trainer.type == 'nodefullbatch_trainer':
        num_train_data = sum(ctx.train_loader.dataset[0]['train_mask'])
    else:
        num_train_data = ctx.num_train_data
    ctx.nbafl_scale_u = cfg.nbafl.w_clip * cfg.federate.total_round_num * \
        cfg.nbafl.constant / num_train_data / \
        cfg.nbafl.epsilon


# ---------------------------------------------------------------------- #
# Additional functions for NbAFL algorithm
# ---------------------------------------------------------------------- #


# Trainer
def _hook_record_initialization(ctx):
    """
    Record the initialized weights within local updates

    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.weight_init``                 Copy from `ctx.model`
        ==================================  ===========================
    """
    ctx.weight_init = deepcopy(
        [_.data.detach() for _ in ctx.model.parameters()])


def _hook_del_initialization(ctx):
    """
    Clear the variable to avoid memory leakage

    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.weight_init``                 Set to `None`
        ==================================  ===========================
    """
    ctx.weight_init = None


def _hook_inject_noise_in_upload(ctx):
    """
    Inject noise into weights before the client upload them to server

    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.model``                       Inject noise to parameters
        ==================================  ===========================
    """
    for p in ctx.model.parameters():
        noise = get_random("Normal", p.shape, {
            "loc": 0,
            "scale": ctx.nbafl_scale_u
        }, p.device)
        p.data += noise


# Server
def inject_noise_in_broadcast(cfg, sample_client_num, model):
    """Inject noise into weights before the server broadcasts them

    """

    # Clip weight
    for p in model.parameters():
        p.data = p.data / torch.max(
            torch.ones(size=p.shape, device=p.data.device),
            torch.abs(p.data) / cfg.nbafl.w_clip)
    if len(sample_client_num) > 0:
        # Inject noise
        L = cfg.federate.sample_client_num if cfg.federate.sample_client_num\
                                              > 0 else cfg.federate.client_num
        if cfg.federate.total_round_num > np.sqrt(cfg.federate.client_num) * L:
            scale_d = 2 * cfg.nbafl.w_clip * cfg.nbafl.constant * np.sqrt(
                np.power(cfg.federate.total_round_num, 2) -
                np.power(L, 2) * cfg.federate.client_num) / (
                    min(sample_client_num) * cfg.federate.client_num *
                    cfg.nbafl.epsilon)
            for p in model.parameters():
                p.data += get_random("Normal", p.shape, {
                    "loc": 0,
                    "scale": scale_d
                }, p.device)


# def wrap_nbafl_server(server: Type[Server]) -> Type[Server]:
def wrap_nbafl_server(server):
    """Register noise injector for the server

    """
    server.register_noise_injector(inject_noise_in_broadcast)
