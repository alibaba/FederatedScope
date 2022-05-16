from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer

import logging

logger = logging.getLogger(__name__)


class GraphMiniBatchTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred

    def _hook_on_batch_forward_flop_count(self, ctx):
        if self.ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                batch = ctx.data_batch.to(ctx.device)
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(ctx.model, batch).total()
                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied by internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, please customize the count hook"
                    )
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except NotImplementedError:
                logger.error(
                    "current flop count implementation is for general NodeFullBatchTrainer case: "
                    "1) the ctx.model takes only batch = ctx.data_batch as input."
                    "Please check the forward format or implement your own flop_count function"
                )

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * ctx.batch_size


def call_graph_level_trainer(trainer_type):
    if trainer_type == 'graphminibatch_trainer':
        trainer_builder = GraphMiniBatchTrainer
        return trainer_builder


register_trainer('graphminibatch_trainer', call_graph_level_trainer)
