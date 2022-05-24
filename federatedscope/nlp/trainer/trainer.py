import logging

from federatedscope.core.monitors import Monitor
from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.auxiliaries import utils

logger = logging.getLogger(__name__)


class NLPTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        x, label = [utils.move_to(_, ctx.device) for _ in ctx.data_batch]
        if isinstance(x, dict):
            pred = ctx.model(**x)[0]
        else:
            pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred

        ctx.batch_size = len(label)

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, this may be caused by "
                f"initializing trainer subclasses without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return

        if self.ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                x, label = [
                    utils.move_to(_, ctx.device) for _ in ctx.data_batch
                ]
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(ctx.model,
                                                    tuple(x.values())).total()

                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied by internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, please customize the count hook"
                    )
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except:
                logger.error(
                    "current flop count implementation is for general NLPTrainer case: "
                    "1) the ctx.model takes only x (for Object) or tuple(x.values()) (for dict) as input."
                    "Please check the forward format or implement your own flop_count function"
                )

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * ctx.batch_size


def call_nlp_trainer(trainer_type):
    if trainer_type == 'nlptrainer':
        trainer_builder = NLPTrainer
        return trainer_builder


register_trainer('nlptrainer', call_nlp_trainer)
