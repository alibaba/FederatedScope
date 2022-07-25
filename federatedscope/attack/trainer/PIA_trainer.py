from typing import Type

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.attack.auxiliary.utils import get_data_property


def wrap_ActivePIATrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    base_trainer.ctx.alpha_prop_loss = base_trainer._cfg.attack.alpha_prop_loss


def hood_on_batch_start_get_prop(ctx):
    ctx.prop = get_data_property(ctx.data_batch)


def hook_on_batch_forward_add_PIA_loss(ctx):
    ctx.loss_batch = ctx.alpha_prop_loss * ctx.loss_batch + (
        1 - ctx.alpha_prop_loss) * ctx.criterion(ctx.y_prob, ctx.prop)
