from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from typing import Type
from copy import deepcopy


def wrap_fedprox_trainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """Implementation of fedprox refer to `Federated Optimization in
    Heterogeneous Networks` [Tian Li, et al., 2020]
        (https://proceedings.mlsys.org/paper/2020/ \
        file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf)

    """

    # ---------------- attribute-level plug-in -----------------------
    init_fedprox_ctx(base_trainer)

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

    return base_trainer


def init_fedprox_ctx(base_trainer):
    """Set proximal regularizer and the factor of regularizer

    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    cfg.defrost()
    cfg.regularizer.type = 'proximal_regularizer'
    cfg.regularizer.mu = cfg.fedprox.mu
    cfg.freeze()

    from federatedscope.core.auxiliaries.regularizer_builder import \
        get_regularizer
    ctx.regularizer = get_regularizer(cfg.regularizer.type)


# ---------------------------------------------------------------------- #
# Additional functions for FedProx algorithm
# ---------------------------------------------------------------------- #


# Trainer
def record_initialization(ctx):
    """Record the initialized weights within local updates

    """
    ctx.weight_init = deepcopy(
        [_.data.detach() for _ in ctx.model.parameters()])


def del_initialization(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.weight_init = None
