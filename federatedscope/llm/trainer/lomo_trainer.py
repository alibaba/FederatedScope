import torch
import logging
from federatedscope.register import register_trainer
# from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.model.adapter_builder import AdapterModel


logger = logging.getLogger(__name__)


class LOMOTrainer(LLMTrainer):
    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # if self.training_args.clip_grad_norm is not None and self.training_args.clip_grad_norm > 0:
        if not ctx.skip_this_batch and ctx.optimizer.clip_grad_norm is not None and ctx.optimizer.clip_grad_norm > 0:
            ctx.optimizer.grad_norm(loss)
            # TODO check how to implement this
            # if ctx.optimizer.loss_scaler and ctx.optimizer.loss_scaler.has_overflow_serial:
            #     # print(f"Gradient overflow, skipping step {self.global_step}")
            #     ctx.optimizer.get_param_coordinator(training=True).reset_step()
            #     # if self.allow_print:
            #     #     self.wandb.log(
            #     #         {
            #     #             'train/loss': loss.item(),
            #     #             'train/learning_rate': self.lr,
            #     #             'train/global_step': self.global_step,
            #     #         },
            #     #         step=self.global_step
            #     #     )

            # else:
            #     ctx.optimizer.get_param_coordinator(training=True).reset_step()
            # 第二次forward
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

            loss = outputs.loss

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        # ctx.optimizer.zero_grad()
        if ctx.skip_this_batch:
            return

        # scaled_loss = loss * self.loss_scaler.loss_scale
        #
        # scaled_loss.backward()
        # # update the last one since the hook function will not be called for the last parameter
        # self.grad_func(0)
        # self.loss_scaler.update_scale(overflow=False)
        ctx.optimizer.fused_backward(ctx.loss_task, ctx.optimizer.lr)
        # TODO check how to implement this
        # ctx.optimizer.get_param_coordinator(training=True).reset_step()
        # ctx.loss_task.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)

        # ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()


def call_lomo_trainer(trainer_type):
    if trainer_type == 'lomotrainer':
        trainer_builder = LOMOTrainer
        return trainer_builder


register_trainer('lomotrainer', call_lomo_trainer)
