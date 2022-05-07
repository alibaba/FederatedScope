from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer


class GraphMAMLTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        # sub task
        batch = ctx.data_batch.to(ctx.device)
        label = batch.y.squeeze(-1).long()
        # inner loop
        if ctx.cur_mode == "train":
            for i in range(5):
                pred = ctx.model(batch)
                ctx.loss_batch = ctx.criterion(pred, label)
        else:
            pred = ctx.model(batch)
            ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_batch.backward(retain_graph=True, create_graph=True)
        ctx.optimizer.step()

        # outer loop for just one task
        batch = ctx.data_batch.to(ctx.device)
        label = batch.y.squeeze(-1).long()

        pred_outer = ctx.model(batch)
        ctx.loss_batch = ctx.criterion(pred_outer, label)
        ctx.optimizer.zero_grad()
        ctx.loss_batch.backward()
        ctx.optimizer.step()




def call_graph_level_trainer(trainer_type):
    if trainer_type == 'graphmaml_trainer':
        trainer_builder = GraphMAMLTrainer
        return trainer_builder


register_trainer('graphmaml_trainer', call_graph_level_trainer)