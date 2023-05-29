import copy
import torch
from accelerate import Accelerator, dispatch_model

from federatedscope.core.trainers.enums import MODE
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self, *args, **kwargs):
        super(LLMTrainer, self).__init__(*args, **kwargs)
        self.use_accelerator = self.ctx.cfg.llm.accelerator.use
        if self.use_accelerator:
            self.accelerator = Accelerator()
            self.device_map = copy.deepcopy(self.ctx.model.hf_device_map)

    def _hook_on_fit_start_init(self, ctx):
        if self.use_accelerator:
            ctx.model = dispatch_model(ctx.model, self.device_map)
        else:
            ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_epoch_start(self, ctx):
        super(LLMTrainer, self)._hook_on_epoch_start(ctx)
        if self.use_accelerator:
            ctx.model, ctx.optimizer, loader = \
                self.accelerator.prepare(ctx.model,
                                         ctx.optimizer,
                                         ctx.get("{}_loader".format(
                                             ctx.cur_split)))
            setattr(ctx, "{}_loader".format(ctx.cur_split), loader)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)

        outputs = ctx.model.forward(input_ids, labels=labels)

        logits = outputs.logits
        loss = outputs.loss

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        if self.use_accelerator:
            # TODO: enable `accelerator.accumulate(model)`
            self.accelerator.backward(ctx.loss_task)
        else:
            ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_fit_end(self, ctx):
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': ctx.loss_batch_total /
            float(ctx.num_samples),
        }
        setattr(ctx, 'eval_metrics', eval_results)


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
