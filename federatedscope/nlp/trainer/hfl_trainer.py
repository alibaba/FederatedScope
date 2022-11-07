import os
import os.path as osp
import copy
import logging
import re
import torch
import numpy as np
import codecs
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import lifecycle, CtxVar
from federatedscope.core.auxiliaries.enums import LIFECYCLE, MODE
from federatedscope.core.auxiliaries.utils import filter_by_specified_keywords
from federatedscope.nlp.monitor.metric_calculator import MetricCalculator, \
    eval_acc
from federatedscope.nlp.trainer.utils import AverageMeter, ContrastiveMonitor
from federatedscope.nlp.dataset.utils import setup_tokenizer
from federatedscope.nlp.dataset.squad import SquadResult
from federatedscope.nlp.dataset.newsqa import NewsQAResult

logger = logging.getLogger(__name__)


class FedNLPTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)
        self.task = config.model.task
        self.ID = None
        self.load_ckpt = True
        self.pred_file, self.src_file, self.tgt_file = None, None, None
        self.finish_eval = False
        self.ctx.eval_metrics = None
        self.ctx.tokenizer = setup_tokenizer(self.cfg)
        self.ctx.grad_accum_count = self.cfg.grad.grad_accum_count
        self.ctx.padding_idx = self.ctx.tokenizer.pad_token_id

    def update_stat(self, ID):
        self.ID = ID
        if self.task in {'cnndm', 'msqg'}:
            pred_dir = osp.join(self.cfg.eval.result_path, 'pred')
            src_dir = osp.join(self.cfg.eval.result_path, 'src')
            tgt_dir = osp.join(self.cfg.eval.result_path, 'tgt')
            self.ctx.pred_path = osp.join(pred_dir, '%d.txt' % ID)
            self.ctx.src_path = osp.join(src_dir, '%d.txt' % ID)
            self.ctx.tgt_path = osp.join(tgt_dir, '%d.txt' % ID)

            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(tgt_dir, exist_ok=True)
            self.pred_file = codecs.open(self.ctx.pred_path, 'w', 'utf-8')
            self.src_file = codecs.open(self.ctx.src_path, 'w', 'utf-8')
            self.tgt_file = codecs.open(self.ctx.tgt_path, 'w', 'utf-8')

    def parse_data(self, data):
        init_dict = dict()
        if isinstance(data, dict):
            for split in ['train', 'val', 'test']:
                init_dict['{}_data'.format(split)] = None
                init_dict['{}_loader'.format(split)] = None
                init_dict['num_{}_data'.format(split)] = 0
                init_dict['{}_encoded'.format(split)] = None
                init_dict['{}_examples'.format(split)] = None
                if data.get(split, None) is not None:
                    if isinstance(data.get(split)['dataloader'], DataLoader):
                        init_dict['{}_loader'.format(split)] = \
                            data.get(split)['dataloader']
                        init_dict['num_{}_data'.format(split)] = len(
                            data.get(split)['dataloader'].dataset)
                        init_dict['{}_encoded'.format(split)] = \
                            data.get(split)['encoded']
                        init_dict['{}_examples'.format(split)] = \
                            data.get(split)['examples']
                    else:
                        raise TypeError('Type {} is not supported.'.format(
                            type(data.get(split))))
        else:
            raise TypeError('Type of data should be dict.')
        return init_dict

    def _load_model(self, ctx):
        load_path = ctx.cfg.federate.hfl_load_from
        global_ckpt_path = osp.join(load_path, 'global_model.pt')
        client_ckpt_path = osp.join(load_path,
                                    'client_model_{}.pt'.format(self.ID))
        if not osp.exists(global_ckpt_path):
            global_dir = os.path.join(load_path, 'global')
            client_dir = os.path.join(load_path, 'client')
            global_ckpt_path = \
                os.path.join(global_dir, 'global_model_{}.pt'.format(self.ID))
            client_ckpt_path = \
                os.path.join(client_dir, 'client_model_{}.pt'.format(self.ID))
            if not osp.exists(global_ckpt_path):
                raise RuntimeError(
                    'Checkpoint NOT found in \'{}\''.format(global_ckpt_path))

        model_ckpt = ctx.model.state_dict()
        logger.info('Loading model from \'{}\''.format(global_ckpt_path))
        global_ckpt = torch.load(global_ckpt_path, map_location='cpu')['model']
        model_ckpt.update({
            k: v
            for k, v in global_ckpt.items()
            if k in model_ckpt and v.size() == model_ckpt[k].size()
        })
        if osp.exists(client_ckpt_path):
            logger.info('Updating model from \'{}\''.format(client_ckpt_path))
            client_ckpt = torch.load(client_ckpt_path,
                                     map_location='cpu')['model']
            model_ckpt.update({
                k: v
                for k, v in client_ckpt.items()
                if k in model_ckpt and v.size() == model_ckpt[k].size()
            })
        ctx.model.load_state_dict(model_ckpt)

    def _save_model(self, ctx):
        if len(ctx.cfg.personalization.local_param) > 0:
            model_ckpt = OrderedDict({
                k: v
                for k, v in ctx.model.state_dict().items()
                if re.search('|'.join(ctx.cfg.personalization.local_param), k)
                is not None
            })
            ckpt = {
                'model': model_ckpt,
                'epoch': ctx.cur_epoch_i + 1,
                'batch': ctx.cur_batch_i + 1,
            }
            save_dir = ctx.cfg.federate.save_to
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = osp.join(save_dir,
                                 'client_model_{}.pt'.format(self.ID))
            torch.save(ckpt, ckpt_path)

    def _remove_special_tokens(self, sent):
        return sent.replace('[CLS]', '').replace('[SEP]', '').\
            replace('[PAD]', '').replace('[unused0]', '').\
            replace('[unused3]', '').replace('[unused1]', ''). \
            replace(r' +', ' ').replace(' [unused2] ', '<q>').\
            replace('[unused2]', '').strip()

    def _run_routine(self, mode, hooks_set, dataset_name=None):
        if self.finish_eval:
            return self.ctx.num_samples
        return super()._run_routine(mode, hooks_set, dataset_name)

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set):
        for batch_i in tqdm(range(
                self.ctx.get(f"num_{self.ctx.cur_split}_batch")),
                            disable=self.ctx.cur_split != "test"):
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_backward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE] and \
                self.ctx.cur_epoch_i == self.ctx.get(
                    f'num_{self.ctx.cur_mode}_epoch') - 1:
                if batch_i >= self.ctx.get(
                        f'num_{self.ctx.cur_mode}_batch_last_epoch') - 1:
                    break

    def _hook_on_fit_start_init(self, ctx):
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.optimizer = ctx.get(f'{ctx.cur_mode}_optimizer')
            ctx.scheduler = ctx.get(f'{ctx.cur_mode}_scheduler')
            if ctx.cfg.federate.hfl_load_from and self.load_ckpt:
                self._load_model(ctx)
                self.load_ckpt = False

        if ctx.cur_split == 'train' and ctx.cfg.federate.hfl_load_from \
                and self.load_ckpt:
            self._load_model(ctx)
            self.load_ckpt = False

        # prepare statistics
        ctx.loss_agg = CtxVar(AverageMeter(), LIFECYCLE.ROUTINE)
        ctx.loss_batch_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.accum_steps = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.squad_results = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.newsqa_results = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        token_ids = ctx.data_batch.get('token_ids', None)
        token_type_ids = ctx.data_batch.get('token_type_ids', None)
        attention_mask = ctx.data_batch.get('attention_mask', None)
        labels = ctx.data_batch.get('labels', None)
        start_positions = ctx.data_batch.get('start_positions', None)
        end_positions = ctx.data_batch.get('end_positions', None)
        example_indices = ctx.data_batch.get('example_indices', None)

        if self.task == 'pretrain':
            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
            )
            ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            pretrain_task = ctx.cfg.model.pretrain_task
            if pretrain_task == 'mlm':
                y_true = labels
            elif pretrain_task == 'denoise':
                y_true = labels[:, 1:]
            count_idx = y_true.ne(-100) & y_true.ne(ctx.padding_idx)
            ctx.y_true = CtxVar(y_true[count_idx], LIFECYCLE.BATCH)
            ctx.y_pred = CtxVar(
                outputs.logits.argmax(dim=-1)[count_idx], LIFECYCLE.BATCH)

        elif self.task in {'imdb', 'agnews'}:
            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                token_type_ids=token_type_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
            )
            ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_pred = CtxVar(outputs.logits.argmax(dim=-1), LIFECYCLE.BATCH)

        elif self.task in {'squad', 'newsqa'}:
            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                token_type_ids=token_type_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                start_positions=start_positions.to(ctx.device),
                end_positions=end_positions.to(ctx.device),
            )
            for i, example_idx in enumerate(example_indices):
                encoded_input = ctx.get('{}_encoded'.format(
                    ctx.cur_split))[example_idx.item()]
                unique_id = int(encoded_input.unique_id)
                start_logits = outputs.logits[0][i].detach().cpu().tolist()
                end_logits = outputs.logits[1][i].detach().cpu().tolist()
                if ctx.cur_split != 'train':
                    if self.task == 'squad':
                        ctx.squad_results.append(
                            SquadResult(unique_id, start_logits, end_logits))
                    elif self.task == 'newsqa':
                        ctx.newsqa_results.append(
                            NewsQAResult(unique_id, start_logits, end_logits))

            ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(torch.cat([start_positions, end_positions]),
                                LIFECYCLE.BATCH)
            ctx.y_pred = CtxVar(
                torch.cat([out.argmax(dim=-1) for out in outputs.logits]),
                LIFECYCLE.BATCH)

        elif self.task in {'cnndm', 'msqg'}:
            if ctx.cur_split != 'test':
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    labels=labels.to(ctx.device),
                )
                ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                y_pred = outputs.logits.argmax(dim=-1)
                y_true = labels[:, 1:]
                non_padding_idx = y_true.ne(ctx.padding_idx)
                ctx.y_true = CtxVar(y_true[non_padding_idx], LIFECYCLE.BATCH)
                ctx.y_pred = CtxVar(y_pred[non_padding_idx], LIFECYCLE.BATCH)

            else:
                outputs = ctx.model.generate(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                )
                # save to file
                out_str = ctx.tokenizer.batch_decode(outputs)
                src_str = ctx.tokenizer.batch_decode(token_ids)
                ref_str = ctx.tokenizer.batch_decode(labels)
                for out, src, ref in zip(out_str, src_str, ref_str):
                    out = self._remove_special_tokens(out)
                    src = self._remove_special_tokens(src)
                    ref = self._remove_special_tokens(ref)
                    self.pred_file.write(out + '\n')
                    self.src_file.write(src + '\n')
                    self.tgt_file.write(ref + '\n')
                self.pred_file.flush()
                self.src_file.flush()
                self.tgt_file.flush()

                ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
                ctx.y_pred = CtxVar(outputs, LIFECYCLE.BATCH)
                ctx.y_true = CtxVar(labels[:, 1:], LIFECYCLE.BATCH)
                return

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

        total_epoch = ctx.get(f'num_{ctx.cur_mode}_epoch')
        total_batch = ctx.get(f'num_{ctx.cur_mode}_batch') if \
            ctx.cur_epoch_i + 1 < total_epoch else ctx.get(
            f'num_{ctx.cur_mode}_batch_last_epoch')
        if ctx.accum_steps == 0:
            if cur_step > 1 and (cur_step % ctx.cfg.trainer.disp_freq == 0
                                 or ctx.cur_batch_i + 1 == total_batch):
                y_true = ctx.y_true.detach().cpu().numpy()
                y_pred = ctx.y_pred.detach().cpu().numpy()
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, axis=-1)
                if y_pred.ndim == 1:
                    y_pred = np.expand_dims(y_pred, axis=-1)
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

            if ctx.cur_batch_i + 1 == total_batch and ctx.cfg.federate.save_to:
                self._save_model(ctx)

    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.get(
            "loss_batch", torch.tensor(0.)).item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

        # cache label for evaluate
        if self.task in {'pretrain', 'squad', 'newsqa', 'cnndm', 'msqg'}:
            ctx.ys_true = CtxVar([ctx.y_true.detach().cpu().numpy()],
                                 LIFECYCLE.ROUTINE)
            ctx.ys_pred = CtxVar([ctx.y_pred.detach().cpu().numpy()],
                                 LIFECYCLE.ROUTINE)
        else:
            ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
            ctx.ys_pred.append(ctx.y_pred.detach().cpu().numpy())

    def _hook_on_fit_end(self, ctx):
        if ctx.cur_split != 'train':
            ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true),
                                 LIFECYCLE.ROUTINE)
            ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred),
                                 LIFECYCLE.ROUTINE)
            results = self.metric_calculator.eval(ctx)
            setattr(ctx, 'eval_metrics', results)

        if ctx.cur_split == 'test' and not self.finish_eval:
            if self.pred_file is not None:
                self.pred_file.close()
            if self.src_file is not None:
                self.src_file.close()
            if self.tgt_file is not None:
                self.tgt_file.close()
            self.finish_eval = True


class PFedNLPTrainer(FedNLPTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.ctx.init_params = copy.deepcopy(model.state_dict())

    def update(self, model_parameters, strict=False):
        super().update(model_parameters, strict=strict)
        self.ctx.init_params = copy.deepcopy(self.ctx.model.state_dict())

    def update_task(self, task):
        self.pretrain_task = task

    def get_model_grads(self, filter_keywords=None):
        if filter_keywords is None:
            filter_keywords = self.ctx.cfg.personalization.local_param
        grads = {}
        for n, p2 in self.ctx.model.state_dict().items():
            if filter_by_specified_keywords(n, filter_keywords):  # preserve
                grads[n] = p2 - self.ctx.init_params[n]
        return grads

    def _load_model(self, ctx):
        load_path = ctx.cfg.federate.hfl_load_from
        global_dir = os.path.join(load_path, 'global')
        client_dir = os.path.join(load_path, 'client')
        global_ckpt_path = os.path.join(global_dir,
                                        'global_model_{}.pt'.format(self.ID))
        client_ckpt_path = os.path.join(client_dir,
                                        'client_model_{}.pt'.format(self.ID))
        if os.path.exists(global_ckpt_path):
            model_ckpt = ctx.model.state_dict()
            logger.info('Loading model from \'{}\''.format(global_ckpt_path))
            global_ckpt = torch.load(global_ckpt_path,
                                     map_location='cpu')['model']
            model_ckpt.update({
                k: v
                for k, v in global_ckpt.items()
                if k in model_ckpt and v.size() == model_ckpt[k].size()
            })
            if os.path.exists(client_ckpt_path):
                logger.info(
                    'Updating model from \'{}\''.format(client_ckpt_path))
                client_ckpt = torch.load(client_ckpt_path,
                                         map_location='cpu')['model']
                model_ckpt.update({
                    k: v
                    for k, v in client_ckpt.items()
                    if k in model_ckpt and v.size() == model_ckpt[k].size()
                })
            ctx.model.load_state_dict(model_ckpt)
        else:
            raise RuntimeError(
                'Checkpoint NOT found in \'{}\''.format(global_ckpt_path))

    def _save_model(self, ctx):
        if len(ctx.cfg.personalization.local_param) > 0:
            model_ckpt = OrderedDict({
                k: v
                for k, v in ctx.model.state_dict().items()
                if re.search('|'.join(ctx.cfg.personalization.local_param), k)
                is not None
            })
            ckpt = {
                'model': model_ckpt,
                'epoch': ctx.cur_epoch_i + 1,
                'batch': ctx.cur_batch_i + 1,
            }
            save_dir = osp.join(ctx.cfg.federate.save_to, 'client')
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = osp.join(save_dir,
                                 'client_model_{}.pt'.format(self.ID))
            torch.save(ckpt, ckpt_path)

    def train(self, target_data_split_name='train', hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return \
            num_samples, self.get_model_para(), self.get_model_grads(), \
            self.ctx.eval_metrics

    def _hook_on_batch_forward(self, ctx):
        if self.task == 'pretrain':
            token_ids = ctx.data_batch[self.pretrain_task]['token_ids']
            attention_mask = \
                ctx.data_batch[self.pretrain_task]['attention_mask']
            labels = ctx.data_batch[self.pretrain_task]['labels']

            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
                pretrain_task=self.pretrain_task,
            )
            ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            if self.pretrain_task == 'mlm':
                y_true = labels
            elif self.pretrain_task == 'denoise':
                y_true = labels[:, 1:]
            count_idx = y_true.ne(-100) & y_true.ne(ctx.padding_idx)
            ctx.y_true = CtxVar(y_true[count_idx], LIFECYCLE.BATCH)
            ctx.y_pred = CtxVar(
                outputs.logits.argmax(dim=-1)[count_idx], LIFECYCLE.BATCH)

        else:
            token_ids = ctx.data_batch.get('token_ids', None)
            token_type_ids = ctx.data_batch.get('token_type_ids', None)
            attention_mask = ctx.data_batch.get('attention_mask', None)
            labels = ctx.data_batch.get('labels', None)
            start_positions = ctx.data_batch.get('start_positions', None)
            end_positions = ctx.data_batch.get('end_positions', None)
            example_indices = ctx.data_batch.get('example_indices', None)

            if self.task in {'imdb', 'agnews'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    labels=labels.to(ctx.device),
                )
                ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
                ctx.y_pred = CtxVar(outputs.logits.argmax(dim=-1),
                                    LIFECYCLE.BATCH)

            elif self.task in {'squad', 'newsqa'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    start_positions=start_positions.to(ctx.device),
                    end_positions=end_positions.to(ctx.device),
                )
                for i, example_idx in enumerate(example_indices):
                    encoded_input = ctx.get('{}_encoded'.format(
                        ctx.cur_split))[example_idx.item()]
                    unique_id = int(encoded_input.unique_id)
                    start_logits = outputs.logits[0][i].detach().cpu().tolist()
                    end_logits = outputs.logits[1][i].detach().cpu().tolist()
                    if ctx.cur_split != 'train':
                        if self.task == 'squad':
                            ctx.squad_results.append(
                                SquadResult(unique_id, start_logits,
                                            end_logits))
                        elif self.task == 'newsqa':
                            ctx.newsqa_results.append(
                                NewsQAResult(unique_id, start_logits,
                                             end_logits))

                ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                ctx.y_true = CtxVar(
                    torch.cat([start_positions, end_positions]),
                    LIFECYCLE.BATCH)
                ctx.y_pred = CtxVar(
                    torch.cat([out.argmax(dim=-1) for out in outputs.logits]),
                    LIFECYCLE.BATCH)

            elif self.task in {'cnndm', 'msqg'}:
                if ctx.cur_split != 'test':
                    outputs = ctx.model(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                        labels=labels.to(ctx.device),
                    )
                    ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
                    ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                    y_pred = outputs.logits.argmax(dim=-1)
                    y_true = labels[:, 1:]
                    non_padding_idx = y_true.ne(ctx.padding_idx)
                    ctx.y_true = CtxVar(y_true[non_padding_idx],
                                        LIFECYCLE.BATCH)
                    ctx.y_pred = CtxVar(y_pred[non_padding_idx],
                                        LIFECYCLE.BATCH)

                else:
                    outputs = ctx.model.generate(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                    )
                    # save to file
                    out_str = ctx.tokenizer.batch_decode(outputs)
                    src_str = ctx.tokenizer.batch_decode(token_ids)
                    ref_str = ctx.tokenizer.batch_decode(labels)
                    for out, src, ref in zip(out_str, src_str, ref_str):
                        out = self._remove_special_tokens(out)
                        src = self._remove_special_tokens(src)
                        ref = self._remove_special_tokens(ref)
                        self.pred_file.write(out + '\n')
                        self.src_file.write(src + '\n')
                        self.tgt_file.write(ref + '\n')
                    self.pred_file.flush()
                    self.src_file.flush()
                    self.tgt_file.flush()

                    ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
                    ctx.y_pred = CtxVar(outputs, LIFECYCLE.BATCH)
                    ctx.y_true = CtxVar(labels[:, 1:], LIFECYCLE.BATCH)
                    return

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

        total_epoch = ctx.get(f'num_{ctx.cur_mode}_epoch')
        total_batch = ctx.get(f'num_{ctx.cur_mode}_batch') if \
            ctx.cur_epoch_i + 1 < total_epoch else ctx.get(
            f'num_{ctx.cur_mode}_batch_last_epoch')
        if ctx.accum_steps == 0:
            if cur_step > 1 and (cur_step % ctx.cfg.trainer.disp_freq == 0
                                 or ctx.cur_batch_i + 1 == total_batch):
                y_true = ctx.y_true.detach().cpu().numpy()
                y_pred = ctx.y_pred.detach().cpu().numpy()
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, axis=-1)
                if y_pred.ndim == 1:
                    y_pred = np.expand_dims(y_pred, axis=-1)
                cur_acc = eval_acc(y_true, y_pred)

                log_str = 'Epoch: [{}/{}][{}/{}]\t' \
                          'LR: {:.2e}\t' \
                          'Acc: {:.4f}\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})'\
                    .format(ctx.cur_epoch_i + 1,
                            total_epoch,
                            cur_step,
                            total_batch // ctx.grad_accum_count,
                            ctx.scheduler.get_last_lr()[0],
                            cur_acc,
                            loss=ctx.loss_agg)

                if ctx.get('regular_loss_batch', None) is not None and \
                        ctx.get('contrast_loss_batch', None) is not None:
                    log_str += \
                        '\tRegular loss: {loss.val:.4f} ' \
                        '({loss.avg:.4f})'.format(loss=ctx.regular_loss_agg)
                    log_str += \
                        '\tContrastive loss: {loss.val:.4f} ' \
                        '({loss.avg:.4f})'.format(
                            loss=ctx.contrast_loss_agg)
                if self.task == 'pretrain':
                    log_str += '\t({})'.format(self.pretrain_task)
                logger.info(log_str)

            if ctx.cur_batch_i + 1 == total_batch and ctx.cfg.federate.save_to:
                self._save_model(ctx)


class PCFedNLPTrainer(PFedNLPTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        self.ctx.contrast_monitor = ContrastiveMonitor()

    def update_contrast_monitor(self, contrast_monitor):
        self.ctx.contrast_monitor = contrast_monitor

    def update_stat(self, ID):
        super().update_stat(ID)
        self.ctx.model.update_client_id(ID)

    @property
    def _in_contrast_prepare(self):
        return self.task != 'pretrain' and self.ctx.cur_split == 'train' and \
               self.ctx.contrast_monitor.stat == 1

    @lifecycle(LIFECYCLE.ROUTINE)
    def _run_routine(self, mode, hooks_set, dataset_name=None):
        raw_num_train_epoch, raw_num_train_batch = None, None
        if self._in_contrast_prepare:
            raw_num_train_epoch, raw_num_train_batch = \
                self.ctx.num_train_epoch, self.ctx.num_train_batch
            batch_size = self.ctx.cfg.data.batch_size
            num_contrast_data = len(self.ctx.contrast_monitor.synth_tokens)
            self.ctx.num_train_epoch = 1
            self.ctx.num_train_batch = \
                num_contrast_data // batch_size + bool(num_contrast_data %
                                                       batch_size)
            self.ctx.num_train_batch_last_epoch = self.ctx.num_train_batch
            self.ctx.num_total_train_batch = \
                self.ctx.num_train_epoch * self.ctx.num_train_batch

        for hook in hooks_set['on_fit_start']:
            hook(self.ctx)

        self._run_epoch(hooks_set)

        for hook in hooks_set["on_fit_end"]:
            hook(self.ctx)

        if raw_num_train_epoch is not None and raw_num_train_batch is not None:
            self.ctx.num_train_epoch = raw_num_train_epoch
            self.ctx.num_train_batch = raw_num_train_batch
            self.ctx.num_train_batch_last_epoch = self.ctx.num_train_batch
            self.ctx.num_total_train_batch = \
                self.ctx.num_train_epoch * self.ctx.num_train_batch

        return self.ctx.num_samples

    @lifecycle(LIFECYCLE.BATCH)
    def _run_batch(self, hooks_set):
        for batch_i in tqdm(range(
                self.ctx.get(f"num_{self.ctx.cur_split}_batch")),
                            disable=not (self._in_contrast_prepare
                                         or self.ctx.cur_split == "test")):
            self.ctx.cur_batch_i = CtxVar(batch_i, LIFECYCLE.BATCH)

            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_backward"]:
                hook(self.ctx)

            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)

            # Break in the final epoch
            if self.ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE] and \
                self.ctx.cur_epoch_i == self.ctx.get(
                    f'num_{self.ctx.cur_mode}_epoch') - 1:
                if batch_i >= self.ctx.get(
                        f'num_{self.ctx.cur_mode}_batch_last_epoch') - 1:
                    break

    def train(self, target_data_split_name='train', hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        return \
            num_samples, self.get_model_para(), self.get_model_grads(), \
            self.ctx.contrast_monitor, self.ctx.eval_metrics

    def parse_data(self, data):
        init_dict = dict()
        if isinstance(data, dict):
            for split in ['train_raw', 'train_contrast', 'val', 'test']:
                init_dict['{}_data'.format(split)] = None
                init_dict['{}_loader'.format(split)] = None
                init_dict['num_{}_data'.format(split)] = 0
                init_dict['{}_encoded'.format(split)] = None
                init_dict['{}_examples'.format(split)] = None
                if data.get(split, None) is not None:
                    if isinstance(data.get(split)['dataloader'], DataLoader):
                        init_dict['{}_loader'.format(split)] = \
                            data.get(split)['dataloader']
                        init_dict['num_{}_data'.format(split)] = \
                            len(data.get(split)['dataloader'].dataset)
                        init_dict['{}_encoded'.format(split)] = \
                            data.get(split)['encoded']
                        init_dict['{}_examples'.format(split)] = \
                            data.get(split)['examples']

                        if split == 'train_raw':
                            init_dict['train_data'] = None
                            init_dict['train_loader'] = \
                                data.get(split)['dataloader']
                            init_dict['num_train_data'] = \
                                len(data.get(split)['dataloader'].dataset)
                            init_dict['train_encoded'] = \
                                data.get(split)['encoded']
                            init_dict['train_examples'] = \
                                data.get(split)['examples']
                    else:
                        raise TypeError('Type {} is not supported.'.format(
                            type(data.get(split))))
        else:
            raise TypeError('Type of data should be dict.')
        return init_dict

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        if self._in_contrast_prepare:
            ctx.train_loader = ctx.train_contrast_loader
        else:
            ctx.regular_loss_agg = CtxVar(AverageMeter(), LIFECYCLE.ROUTINE)
            ctx.contrast_loss_agg = CtxVar(AverageMeter(), LIFECYCLE.ROUTINE)
            ctx.train_loader = ctx.train_raw_loader

    def _hook_on_batch_forward(self, ctx):
        ctx.contrast_loss_batch = CtxVar(None, LIFECYCLE.BATCH)
        if self.task == 'pretrain':
            token_ids = ctx.data_batch[self.pretrain_task]['token_ids']
            attention_mask = \
                ctx.data_batch[self.pretrain_task]['attention_mask']
            labels = ctx.data_batch[self.pretrain_task]['labels']
            example_indices = \
                ctx.data_batch[self.pretrain_task]['example_indices']

            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
                pretrain_task=self.pretrain_task,
                example_indices=example_indices,
            )
            ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            if self.pretrain_task == 'mlm':
                y_true = labels
            elif self.pretrain_task == 'denoise':
                y_true = labels[:, 1:]
            count_idx = y_true.ne(-100) & y_true.ne(ctx.padding_idx)
            ctx.y_true = CtxVar(y_true[count_idx], LIFECYCLE.BATCH)
            ctx.y_pred = CtxVar(
                outputs.logits.argmax(dim=-1)[count_idx], LIFECYCLE.BATCH)

        else:
            token_ids = ctx.data_batch.get('token_ids', None)
            token_type_ids = ctx.data_batch.get('token_type_ids', None)
            attention_mask = ctx.data_batch.get('attention_mask', None)
            labels = ctx.data_batch.get('labels', None)
            start_positions = ctx.data_batch.get('start_positions', None)
            end_positions = ctx.data_batch.get('end_positions', None)
            example_indices = ctx.data_batch.get('example_indices', None)

            if self.task in {'imdb', 'agnews'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    labels=labels.to(ctx.device),
                    contrast_monitor=ctx.contrast_monitor,
                    in_contrast_prepare=self._in_contrast_prepare,
                    example_indices=example_indices,
                )
                if not self._in_contrast_prepare:
                    ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
                    ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                    ctx.regular_loss_batch = CtxVar(outputs.regular_loss,
                                                    LIFECYCLE.BATCH)
                    ctx.contrast_loss_batch = CtxVar(outputs.contrast_loss,
                                                     LIFECYCLE.BATCH)
                    ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
                    ctx.y_pred = CtxVar(outputs.logits.argmax(dim=-1),
                                        LIFECYCLE.BATCH)

            elif self.task in {'squad', 'newsqa'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    start_positions=start_positions.to(ctx.device),
                    end_positions=end_positions.to(ctx.device),
                    contrast_monitor=ctx.contrast_monitor,
                    in_contrast_prepare=self._in_contrast_prepare,
                    example_indices=example_indices,
                )
                if not self._in_contrast_prepare:
                    for i, example_idx in enumerate(example_indices):
                        encoded_input = ctx.get('{}_encoded'.format(
                            ctx.cur_split))[example_idx.item()]
                        unique_id = int(encoded_input.unique_id)
                        start_logits = \
                            outputs.logits[0][i].detach().cpu().tolist()
                        end_logits = \
                            outputs.logits[1][i].detach().cpu().tolist()
                        if ctx.cur_split != 'train':
                            if self.task == 'squad':
                                ctx.squad_results.append(
                                    SquadResult(unique_id, start_logits,
                                                end_logits))
                            elif self.task == 'newsqa':
                                ctx.newsqa_results.append(
                                    NewsQAResult(unique_id, start_logits,
                                                 end_logits))

                    ctx.batch_size = CtxVar(len(token_ids), LIFECYCLE.BATCH)
                    ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                    ctx.regular_loss_batch = CtxVar(outputs.regular_loss,
                                                    LIFECYCLE.BATCH)
                    ctx.contrast_loss_batch = CtxVar(outputs.contrast_loss,
                                                     LIFECYCLE.BATCH)
                    ctx.y_true = CtxVar(
                        torch.cat([start_positions, end_positions]),
                        LIFECYCLE.BATCH)
                    ctx.y_pred = CtxVar(
                        torch.cat(
                            [out.argmax(dim=-1) for out in outputs.logits]),
                        LIFECYCLE.BATCH)

            elif self.task in {'cnndm', 'msqg'}:
                if ctx.cur_split != 'test':
                    outputs = ctx.model(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                        labels=labels.to(ctx.device),
                        contrast_monitor=ctx.contrast_monitor,
                        in_contrast_prepare=self._in_contrast_prepare,
                        example_indices=example_indices,
                    )
                    if not self._in_contrast_prepare:
                        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
                        ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
                        ctx.regular_loss_batch = CtxVar(
                            outputs.regular_loss, LIFECYCLE.BATCH)
                        ctx.contrast_loss_batch = CtxVar(
                            outputs.contrast_loss, LIFECYCLE.BATCH)

                        y_pred = outputs.logits.argmax(dim=-1)
                        y_true = labels[:, 1:]
                        non_padding_idx = y_true.ne(ctx.padding_idx)
                        ctx.y_true = CtxVar(y_true[non_padding_idx],
                                            LIFECYCLE.BATCH)
                        ctx.y_pred = CtxVar(y_pred[non_padding_idx],
                                            LIFECYCLE.BATCH)
                else:
                    outputs = ctx.model.generate(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                    )
                    # save to file
                    out_str = ctx.tokenizer.batch_decode(outputs)
                    src_str = ctx.tokenizer.batch_decode(token_ids)
                    ref_str = ctx.tokenizer.batch_decode(labels)
                    for out, src, ref in zip(out_str, src_str, ref_str):
                        out = self._remove_special_tokens(out)
                        src = self._remove_special_tokens(src)
                        ref = self._remove_special_tokens(ref)
                        self.pred_file.write(out + '\n')
                        self.src_file.write(src + '\n')
                        self.tgt_file.write(ref + '\n')
                    self.pred_file.flush()
                    self.src_file.flush()
                    self.tgt_file.flush()

                    ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
                    ctx.y_pred = CtxVar(outputs, LIFECYCLE.BATCH)
                    ctx.y_true = CtxVar(labels[:, 1:], LIFECYCLE.BATCH)
                    return

        if self._in_contrast_prepare:
            ctx.batch_size = CtxVar(0, LIFECYCLE.BATCH)
            dec_out, dec_hidden, example_indices = \
                outputs.logits, outputs.hidden_states, outputs.example_indices
            if len(example_indices) > 0:
                for ex, out in zip(example_indices, dec_out.detach().cpu()):
                    ctx.contrast_monitor.update_dec_out(out, k=ex.item())
                for ex, hids in zip(example_indices,
                                    dec_hidden.detach().cpu()):
                    ctx.contrast_monitor.update_dec_hidden(hids, k=ex.item())
        else:
            ctx.loss_agg.update(ctx.loss_batch.detach().item(), ctx.batch_size)
            if ctx.get('regular_loss_batch', None) is not None:
                ctx.regular_loss_agg.update(
                    ctx.regular_loss_batch.detach().item(), ctx.batch_size)
            if ctx.get('contrast_loss_batch', None) is not None:
                ctx.contrast_loss_agg.update(
                    ctx.contrast_loss_batch.detach().item(), ctx.batch_size)

    def _hook_on_batch_forward_regularizer(self, ctx):
        if self._in_contrast_prepare:
            return
        super()._hook_on_batch_forward_regularizer(ctx)

    def _hook_on_batch_backward(self, ctx):
        if self._in_contrast_prepare:
            return
        super()._hook_on_batch_backward(ctx)

    def _hook_on_batch_end(self, ctx):
        if self._in_contrast_prepare:
            return
        super()._hook_on_batch_end(ctx)

    def _hook_on_fit_end(self, ctx):
        if self.task != 'pretrain' and ctx.cur_split == 'train':
            ctx.contrast_monitor.update_stat(ctx.contrast_monitor.stat + 1)
            return
        super()._hook_on_fit_end(ctx)
