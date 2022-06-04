from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer

import learn2learn as l2l

class GraphMAMLTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        maml = l2l.algorithms.MAML(self.ctx.model, lr=self.cfg.maml.inner_lr)
        ctx.clone = maml.clone()

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        label = batch.y.squeeze(-1).long()

        if ctx.get("finetune", False):
            # update on the model
            pred = ctx.model(batch)
        else:
            # update on the clone
            pred = ctx.clone(batch)
        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred

    def _hook_on_batch_backward(self, ctx):
        if ctx.get("finetune", False):
            # normal update when finetune
            ctx.optimizer.zero_grad()
            ctx.loss_batch.backward()
            ctx.optimizer.step()
        else:
            # during train, fake forward
            ctx.clone.adapt(ctx.loss_batch, allow_unused=True)

    def _hook_on_batch_end(self, ctx):
        # keep the last batch here
        data_batch = ctx.data_batch
        super()._hook_on_batch_end(ctx)
        ctx.data_batch = data_batch

    def _hook_on_fit_end(self, ctx):
        if ctx.cur_mode == "train" and not ctx.get("finetune", False):
            # outer loop, reuse the last batch
            batch = ctx.data_batch.to(ctx.device)
            label = batch.y.squeeze(-1).long()
            # forward
            pred_outer = ctx.clone(batch)
            ctx.loss_batch = ctx.criterion(pred_outer, label)

            ctx.optimizer.zero_grad()
            ctx.loss_batch.backward()
            ctx.optimizer.step()

        ctx.data_batch = None
        ctx.clone = None
        super()._hook_on_fit_end(ctx)


def call_graph_level_trainer(trainer_type):
    if trainer_type == 'graphmaml_trainer':
        trainer_builder = GraphMAMLTrainer
        return trainer_builder


register_trainer('graphmaml_trainer', call_graph_level_trainer)