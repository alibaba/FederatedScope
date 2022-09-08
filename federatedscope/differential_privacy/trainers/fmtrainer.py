from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.trainers.context import CtxVar


class FMTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred, loss = ctx.model(x, label)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
