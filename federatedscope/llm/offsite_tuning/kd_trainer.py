import torch
import logging
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

logger = logging.getLogger(__name__)


def get_kd_loss(raw_model, adap_model):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
    kwargs = adap_model.student_l.input_kwargs
    args = adap_model.student_l.input_args
    output_teacher = args[0]
    args = list(args[1:])
    args = tuple(args)

    with torch.no_grad():
        raw_model.teacher.eval()
        for teacher_layer in raw_model.teacher:
            output_teacher = teacher_layer(output_teacher, *args, **kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]

    output_student = adap_model.student_r.cached_output.float()
    output_teacher = output_teacher.float()

    std = output_teacher.pow(2).mean().sqrt()
    kd_loss = (output_teacher - output_student).div(std).pow(2).mean()
    return kd_loss


class KDTrainer(LLMTrainer):
    def __init__(self,
                 raw_model,
                 adapter_model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(KDTrainer, self).__init__(adapter_model, data, device, config,
                                        only_for_eval, monitor)
        self.ctx.raw_model = raw_model.to(device)
        self.lm_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.lm_loss_weight
        self.kd_loss_weight = \
            config.llm.offsite_tuning.emu_align.train.kd_loss_weight

    def _hook_on_fit_start_numerical_precision(self, ctx):
        super(KDTrainer, self)._hook_on_fit_start_numerical_precision(ctx)
        if self.cfg.train.is_enable_half:
            ctx.raw_model = ctx.raw_model.half()

    def train(self, target_data_split_name="train", hooks_set=None):
        num_samples, model_para_all, eval_metrics = \
            super(KDTrainer, self).train(target_data_split_name, hooks_set)
        logger.info("Finish alignment, move raw model to cpu.")
        self.ctx.raw_model.cpu()
        return num_samples, model_para_all, eval_metrics

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask)

        logits = outputs.logits
        kd_loss = self.kd_loss_weight * get_kd_loss(ctx.raw_model, ctx.model)
        lm_loss = self.lm_loss_weight * outputs.loss
        loss = kd_loss + lm_loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        logger.info(f'lm_loss: {lm_loss.item()}, kd loss: {kd_loss.item()}')
