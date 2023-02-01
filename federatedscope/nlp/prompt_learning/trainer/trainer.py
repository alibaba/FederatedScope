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
        self.ctx.eval_metrics = None
        self.ctx.tokenizer = setup_tokenizer(config)
        self.ctx.grad_accum_count = config.grad.grad_accum_count
        self.ctx.init_params = copy.deepcopy(model.state_dict())

    def update(self, model_parameters, strict=False):
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        merged_param = merge_param_dict(self.ctx.model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.ctx.model.load_state_dict(merged_param, strict=strict)
        self.ctx.init_params = copy.deepcopy(self.ctx.model.state_dict())

    def get_model_grads(self, filter_keywords=None):
        if filter_keywords is None:
            filter_keywords = self.ctx.cfg.personalization.local_param
        grads = {}
        for n, p2 in self.ctx.model.state_dict().items():
            if filter_by_specified_keywords(n, filter_keywords):  # preserve
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
        optimizer = get_optimizer(ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
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

    def _hook_on_batch_forward(self, ctx):
        token_ids = ctx.data_batch['input_ids']
        token_type_ids = ctx.data_batch['token_type_ids']
        attention_mask = ctx.data_batch['attention_mask']
        labels = ctx.data_batch['labels']

        outputs = ctx.model(
            input_ids=token_ids.to(ctx.device),
            token_type_ids=token_type_ids.to(ctx.device),
            attention_mask=attention_mask.to(ctx.device),
            labels=labels.to(ctx.device),
        )
        ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
        ctx.loss_agg.update(ctx.loss_batch.detach().item(), ctx.batch_size)

    def _hook_on_batch_backward(self, ctx):
        cur_step = (ctx.cur_batch_i + 1) // ctx.grad_accum_count
        ctx.accum_steps += 1
        ctx.loss_task /= ctx.grad_accum_count
        ctx.loss_task.backward()

        if ctx.accum_steps == ctx.grad_accum_count:
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
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
                y_prob = ctx.y_prob.detach().cpu().numpy()
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, axis=-1)
                if y_prob.ndim == 2:
                    y_prob = np.expand_dims(y_prob, axis=-1)
                y_pred = np.argmax(y_prob, axis=1)
                cur_acc = eval_acc(y_true, y_pred)

                logger.info('Epoch: [{}/{}][{}/{}]\t'
                            'LR: {:.2e}\t'
                            'Acc: {:.4f}\t'
                            'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                                ctx.cur_epoch_i + 1,
                                total_epoch,
                                cur_step,
                                total_batch // ctx.grad_accum_count,
                                ctx.scheduler.get_last_lr()[0],
                                cur_acc,
                                loss=ctx.loss_agg))
