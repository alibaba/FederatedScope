import copy
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.auxiliaries.utils import param2tensor
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.core.trainers.utils import filter_by_specified_keywords
from federatedscope.core.monitors.metric_calculator import eval_acc
from federatedscope.nlp.prompt_learning.trainer.utils import AverageMeter, \
    merge_param_dict
from federatedscope.nlp.prompt_learning.dataset.utils import setup_tokenizer

logger = logging.getLogger(__name__)


class PLTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        self.role = self.ctx.model.role
        self.personalized_param = self.ctx.cfg.personalization.get(
            f'{self.ctx.model.role}_local_param')
        self.ctx.eval_metrics = None
        self.ctx.tokenizer = setup_tokenizer(config)
        self.ctx.grad_accum_count = config.grad.grad_accum_count
        self.ctx.init_params = copy.deepcopy(model.state_dict())
        self.do_alter_train = \
            self.role == 'client' and self.ctx.cfg.federate.pl_alter_train
        self.alter_stage = None
        self.alter_model_param = self.ctx.cfg.model.alter_model_param
        self.alter_prompt_param = self.ctx.cfg.model.alter_prompt_param

    @property
    def use_kd_loss(self):
        use_init_kd_loss = \
            self.ctx.cfg.federate.pl_init_kd and self.role == 'client'
        use_c2s_kd_loss = \
            self.ctx.cfg.model.use_c2s_kd_loss and self.role == 'server'
        use_s2c_kd_loss = \
            self.ctx.cfg.model.use_s2c_kd_loss and self.role == 'client'
        return \
            self.ctx.cur_mode == 'train' and \
            self.ctx.get('teacher_model', None) and \
            (use_init_kd_loss or use_c2s_kd_loss or use_s2c_kd_loss)

    def update(self, model_parameters, strict=False):
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        merged_param = merge_param_dict(
            self.ctx.model.state_dict().copy(),
            self._param_filter(model_parameters, self.personalized_param))
        self.ctx.model.load_state_dict(merged_param, strict=strict)
        self.ctx.init_params = copy.deepcopy(self.ctx.model.state_dict())

    def update_alter_stage(self, stage):
        assert stage in {'model', 'prompt'}
        self.alter_stage = stage
        if stage == 'model':
            self.ctx.model._freeze_param(self.alter_prompt_param)
            self.ctx.model._unfreeze_param(self.alter_model_param)
        else:
            self.ctx.model._freeze_param(self.alter_model_param)
            self.ctx.model._unfreeze_param(self.alter_prompt_param)

    def get_model_para(self):
        return self._param_filter(
            self.ctx.model.state_dict() if self.cfg.federate.share_local_model
            else self.ctx.model.cpu().state_dict(), self.personalized_param)

    def get_model_grads(self):
        grads = {}
        for n, p2 in self.ctx.model.state_dict().items():
            # preserve
            if filter_by_specified_keywords(n, self.personalized_param):
                grads[n] = p2 - self.ctx.init_params[n]
        return grads

    def parse_data(self, data):
        init_dict = dict()
        if isinstance(data, dict):
            all_split = ['train', 'val', 'test']
            for split in all_split:
                init_dict['{}_data'.format(split)] = None
                init_dict['{}_loader'.format(split)] = None
                init_dict['num_{}_data'.format(split)] = 0
                if data.get(split, None) is not None:
                    if isinstance(data.get(split)['dataloader'], DataLoader):
                        init_dict['{}_loader'.format(split)] = \
                            data.get(split)['dataloader']
                        init_dict['num_{}_data'.format(split)] = \
                            len(data.get(split)['dataloader'].dataset)
                    else:
                        raise TypeError('Type {} is not supported.'.format(
                            type(data.get(split))))
        else:
            raise TypeError('Type of data should be dict.')

        return init_dict

    def setup_optimizer_and_scheduler(self, ctx):
        total_steps = getattr(ctx, f'num_total_{ctx.cur_mode}_batch',
                              None) // ctx.cfg.grad.grad_accum_count * \
                      ctx.cfg.federate.total_round_num
        warmup_steps = int(ctx.cfg[ctx.cur_mode].scheduler.warmup_ratio *
                           total_steps)
        if self.do_alter_train:
            optimizer, scheduler = {}, {}
            for stage in ['model', 'prompt']:
                params = [
                    p for n, p in self.ctx.model.named_parameters()
                    if not filter_by_specified_keywords(
                        n, getattr(self, f'alter_{stage}_param'))
                ]
                optimizer[stage] = get_optimizer(
                    params, **ctx.cfg[ctx.cur_mode].optimizer)
                scheduler[stage] = get_scheduler(optimizer[stage],
                                                 **ctx.cfg.train.scheduler,
                                                 total_steps=total_steps,
                                                 warmup_steps=warmup_steps)
        else:
            optimizer = get_optimizer(ctx.model,
                                      **ctx.cfg[ctx.cur_mode].optimizer)
            scheduler = get_scheduler(optimizer,
                                      **ctx.cfg.train.scheduler,
                                      total_steps=total_steps,
                                      warmup_steps=warmup_steps)

        return optimizer, scheduler

    def train(self, target_data_split_name='train', hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return \
            num_samples, self.get_model_para(), self.get_model_grads(), \
            self.ctx.eval_metrics

    def _hook_on_fit_start_init(self, ctx):
        ctx.model.to(ctx.device)
        if ctx.get('teacher_model', None):
            ctx.teacher_model.to(ctx.device)
            ctx.teacher_model.eval()

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.optimizer = ctx.get(f'{ctx.cur_mode}_optimizer', None)
            ctx.scheduler = ctx.get(f'{ctx.cur_mode}_scheduler', None)
            if ctx.optimizer is None or ctx.scheduler is None:
                ctx.optimizer, ctx.scheduler = \
                    self.setup_optimizer_and_scheduler(ctx)
                setattr(ctx, f'{ctx.cur_mode}_optimizer', ctx.optimizer)
                setattr(ctx, f'{ctx.cur_mode}_scheduler', ctx.scheduler)

        # prepare statistics
        ctx.loss_agg = CtxVar(AverageMeter(), LIFECYCLE.ROUTINE)
        ctx.loss_batch_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.accum_steps = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        if self.use_kd_loss:
            ctx.regular_loss_agg = CtxVar(AverageMeter(), LIFECYCLE.ROUTINE)
            ctx.kd_loss_agg = CtxVar(AverageMeter(), LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        token_ids = ctx.data_batch['input_ids']
        attention_mask = ctx.data_batch['attention_mask']
        labels = ctx.data_batch['labels']

        outputs = ctx.model(input_ids=token_ids.to(ctx.device),
                            attention_mask=attention_mask.to(ctx.device),
                            labels=labels.to(ctx.device),
                            use_kd_loss=self.use_kd_loss,
                            teacher_model=ctx.get('teacher_model', None))
        ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        # ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(outputs.logits.argmax(dim=-1), LIFECYCLE.BATCH)
        ctx.loss_agg.update(ctx.loss_batch.detach().item(), ctx.batch_size)
        if self.use_kd_loss:
            ctx.regular_loss_batch = CtxVar(outputs.regular_loss,
                                            LIFECYCLE.BATCH)
            ctx.kd_loss_batch = CtxVar(outputs.kd_loss, LIFECYCLE.BATCH)
            ctx.regular_loss_agg.update(ctx.regular_loss_batch.detach().item(),
                                        ctx.batch_size)
            ctx.kd_loss_agg.update(ctx.kd_loss_batch.detach().item(),
                                   ctx.batch_size)

    def _hook_on_batch_backward(self, ctx):
        cur_step = (ctx.cur_batch_i + 1) // ctx.grad_accum_count
        ctx.accum_steps += 1
        ctx.loss_task /= ctx.grad_accum_count
        ctx.loss_task.backward()

        if ctx.accum_steps == ctx.grad_accum_count:
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
            if self.do_alter_train:
                ctx.optimizer[self.alter_stage].step()
                ctx.scheduler[self.alter_stage].step()
                ctx.optimizer[self.alter_stage].zero_grad()
            else:
                ctx.optimizer.step()
                ctx.scheduler.step()
                ctx.optimizer.zero_grad()
            ctx.accum_steps = CtxVar(0, LIFECYCLE.ROUTINE)

        total_epoch = getattr(ctx, f'num_{ctx.cur_mode}_epoch', None)
        total_batch = getattr(ctx, f'num_{ctx.cur_mode}_batch', None) if \
            ctx.cur_epoch_i + 1 < total_epoch else \
            getattr(ctx, f'num_{ctx.cur_mode}_batch_last_epoch', None)
        if ctx.accum_steps == 0:
            if (cur_step > 1 and cur_step % ctx.cfg.trainer.disp_freq == 0) \
                    or ctx.cur_batch_i + 1 == total_batch:
                y_true = ctx.y_true.detach().cpu().numpy()
                y_pred = ctx.y_pred.detach().cpu().numpy()
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, axis=-1)
                if y_pred.ndim == 1:
                    y_pred = np.expand_dims(y_pred, axis=-1)
                cur_acc = eval_acc(y_true, y_pred)

                lr = ctx.scheduler[self.alter_stage].get_last_lr()[0] if \
                    self.do_alter_train else ctx.scheduler.get_last_lr()[0]
                log_str = '({}) '.format(self.alter_stage) if \
                    self.do_alter_train else ''
                log_str += 'Epoch: [{}/{}][{}/{}]\t'\
                           'LR: {:.2e}\t'\
                           'Acc: {:.4f}\t'\
                           'Loss: {loss.val:.4f} ({loss.avg:.4f})'\
                    .format(ctx.cur_epoch_i + 1,
                            total_epoch,
                            cur_step,
                            total_batch // ctx.grad_accum_count,
                            lr,
                            cur_acc,
                            loss=ctx.loss_agg)
                if self.use_kd_loss:
                    log_str += \
                        '\tRegular loss: {loss.val:.4f} ' \
                        '({loss.avg:.4f})'.format(loss=ctx.regular_loss_agg)
                    log_str += \
                        '\tKD loss: {loss.val:.4f} ' \
                        '({loss.avg:.4f})'.format(loss=ctx.kd_loss_agg)
                logger.info(log_str)

    def _hook_on_batch_end(self, ctx):
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        ctx.ys_true = CtxVar([ctx.y_true.detach().cpu().numpy()],
                             LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([ctx.y_pred.detach().cpu().numpy()],
                             LIFECYCLE.ROUTINE)

    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        setattr(ctx, 'eval_metrics', results)
        if ctx.get('teacher_model', None):
            ctx.teacher_model.train()

    def discharge_model(self):
        super().discharge_model()
        if self.ctx.get('teacher_model', None):
            self.ctx.teacher_model.to(torch.device("cpu"))
