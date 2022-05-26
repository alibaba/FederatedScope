from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer


class GraphMiniBatchTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        if self.cfg.model.task.endswith('Regression'):
            label = batch.y.float()
        else:
            label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


def call_graph_level_trainer(trainer_type):
    if trainer_type == 'graphminibatch_trainer':
        trainer_builder = GraphMiniBatchTrainer
        return trainer_builder


register_trainer('graphminibatch_trainer', call_graph_level_trainer)