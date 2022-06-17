import os
import os.path as osp
import collections
import copy
import logging
import re
import torch
import numpy as np
import learn2learn as l2l
from collections import OrderedDict
from torch.utils.data import DataLoader
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.monitors.metric_calculator import MetricCalculator, eval_acc
from federatedscope.register import register_trainer
from federatedscope.nlp.metrics.sts import compute_sts_metrics
from federatedscope.nlp.trainer.context import MyContext
from federatedscope.nlp.auxiliaries.utils import AverageMeter
from federatedscope.nlp.dataset.squad import SquadResult

logger = logging.getLogger(__name__)


# Build your trainer here.
class MAMLTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False):
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)

        self.ctx = MyContext(model=model,
                             cfg=self.cfg,
                             data=data,
                             device=device,
                             init_dict=self.parse_data(data))

        # Atomic operation during training/evaluation
        self.hooks_in_train = collections.defaultdict(list)

        # By default, use the same trigger keys
        self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)

        # register necessary hooks into self.hooks_in_train and self.hooks_in_eval
        if not only_for_eval:
            self.register_default_hooks_train()
        self.register_default_hooks_eval()

        if self.cfg.federate.mode == 'distributed':
            self.print_trainer_meta_info()
        else:
            # in standalone mode, by default, we print the trainer info only once for better logs readability
            pass

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader", "num_{}_data", "{}_encoded", "{}_examples" for different modes
        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                init_dict["{}_encoded".format(mode)] = None
                init_dict["{}_examples".format(mode)] = None
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode)['dataloader'], DataLoader):
                        init_dict["{}_loader".format(mode)] = data.get(mode)['dataloader']
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode)['dataloader'].dataset)
                        init_dict["{}_encoded".format(mode)] = data.get(mode)['encoded']
                        init_dict["{}_examples".format(mode)] = data.get(mode)['examples']
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def _store_ctx(self, ctx):
        store_dict = {}
        store_dict['data_batch'] = ctx.data_batch
        store_dict['batch_size'] = ctx.batch_size
        store_dict['loss_task'] = ctx.loss_task
        store_dict['loss_batch'] = ctx.loss_batch
        store_dict['loss_regular'] = ctx.loss_regular
        store_dict['y_true'] = ctx.y_true
        store_dict['y_prob'] = ctx.y_prob
        return store_dict

    def _restore_ctx(self, ctx, store_dict):
        for k, v in store_dict.items():
            setattr(ctx, k, v)

    def _load_model(self, ctx):
        load_path = ctx.cfg.federate.load_from
        global_ckpt_path = osp.join(load_path, 'global_model.pt')
        client_ckpt_path = osp.join(load_path, 'client_model_{}.pt'.format(ctx.cfg.data.type))
        if osp.exists(global_ckpt_path) and osp.exists(client_ckpt_path):
            logger.info('Loading model from \'{}\''.format(load_path))
            global_ckpt = torch.load(global_ckpt_path, map_location='cpu')['model']
            client_ckpt = torch.load(client_ckpt_path, map_location='cpu')['model']
            global_ckpt.update(client_ckpt)
            model_ckpt = global_ckpt
            ctx.model.load_state_dict(model_ckpt)

    def _test(self, ctx):
        logger.info('==> Start test evaluation')
        store_ctx = self._store_ctx(ctx)
        test_metrics = self.evaluate('test')
        logger.info('Test metrics before aggregation: {}'.format(test_metrics))
        self._restore_ctx(ctx, store_ctx)

    def _save_model(self, ctx):
        model_ckpt = OrderedDict({k: v for k, v in ctx.model.state_dict().items()
                                  if re.search('|'.join(ctx.cfg.personalization.local_param), k) is not None})
        ckpt = {
            'model': model_ckpt,
            'epoch': ctx.cur_epoch_i + 1,
            'batch': ctx.cur_batch_i + 1,
        }
        save_dir = '/'.join(osp.normpath(ctx.cfg.federate.save_to).split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = osp.join(save_dir, 'client_model_{}.pt'.format(ctx.cfg.data.type))
        torch.save(ckpt, ckpt_path)

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)
        if ctx.cur_data_split == 'train' and ctx.cfg.federate.load_from:
            self._load_model(ctx)
        setattr(ctx, "loss_func", getattr(ctx, "{}_loss_func".format(ctx.cur_data_split), None))
        if ctx.loss_func is not None:
            ctx.loss_func.to(ctx.device)

        # prepare statistics
        setattr(ctx, "loss_agg_{}".format(ctx.cur_data_split), AverageMeter())
        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_squad_results".format(ctx.cur_data_split), [])
        setattr(ctx, 'accum_steps', 0)

        maml = l2l.algorithms.MAML(ctx.model, lr=ctx.cfg.maml.inner_lr)
        ctx.maml = maml.clone()

    def _hook_on_batch_forward(self, ctx):
        task = ctx.cfg.data.type
        if task in {'sts', 'imdb'}:
            token_ids, token_type_ids, attention_mask, labels = [_.to(ctx.device) for _ in ctx.data_batch]
            outputs = ctx.maml(
                input_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                config=ctx.cfg,
            )

            ctx.batch_size = len(token_ids)
            ctx.loss_batch = outputs.loss
            ctx.y_true = labels
            ctx.y_prob = outputs.logits

        elif task == 'squad':
            token_ids, token_type_ids, attention_mask, start_positions, end_positions, example_indices = \
                [_.to(ctx.device) for _ in ctx.data_batch]
            outputs = ctx.maml(
                input_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                config=ctx.cfg,
            )

            for i, example_idx in enumerate(example_indices):
                encoded_input = ctx.get('{}_encoded'.format(ctx.cur_data_split))[example_idx.item()]
                unique_id = int(encoded_input.unique_id)
                start_logits = outputs.logits[0][i].detach().cpu().tolist()
                end_logits = outputs.logits[1][i].detach().cpu().tolist()
                ctx.get('{}_squad_results'.format(ctx.cur_data_split)).append(
                        SquadResult(unique_id, start_logits, end_logits))

            ctx.batch_size = len(token_ids)
            ctx.loss_batch = outputs.loss
            ctx.y_true = torch.cat([start_positions, end_positions])
            ctx.y_prob = torch.cat(outputs.logits)

        ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(ctx.loss_batch.detach().item(), ctx.batch_size)

    def _hook_on_batch_backward(self, ctx):
        cur_step = ctx.cur_batch_i + 1
        cur_task = ctx.cfg.data.type

        ctx.maml.adapt(ctx.loss_task, allow_nograd=True, allow_unused=True)

        if cur_step > 0 and ctx.accum_steps == 0:
            if cur_step > 1:
                if cur_step % ctx.cfg.trainer.disp_freq == 0 or ctx.cur_batch_i + 1 == ctx.num_train_batch:
                    if cur_task == 'sts':
                        y_true = ctx.y_true.detach().cpu().numpy()
                        y_pred = ctx.y_prob.detach().cpu().numpy()
                        total_y_true = np.concatenate(ctx.get('{}_y_true'.format(ctx.cur_data_split)))
                        total_y_pred = np.concatenate(ctx.get('{}_y_prob'.format(ctx.cur_data_split)))

                        if len(y_true) > 1: y_true = y_true.squeeze()
                        if len(y_pred) > 1: y_pred = y_pred.squeeze()
                        if len(total_y_pred) > 1: total_y_true = total_y_true.squeeze()
                        if len(total_y_pred) > 1: total_y_pred = total_y_pred.squeeze()

                        logger.info('(Inner) Epoch: [{}/{}][{}/{}]\t'
                                    'LR: {:.2e}\t'
                                    'Corr: {:.4f} ({:.4f})\t'
                                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                                    .format(ctx.cur_epoch_i + 1,
                                            ctx.num_train_epoch,
                                            cur_step,
                                            ctx.cfg.trainer.train_steps,
                                            ctx.cfg.maml.inner_lr,
                                            compute_sts_metrics(y_pred, y_true)['corr'] if len(y_pred) > 1 else 0,
                                            compute_sts_metrics(total_y_pred, total_y_true)['corr'] if len(total_y_pred) > 1 else 0,
                                            loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split))))

                    elif cur_task in {'imdb', 'squad'}:
                        y_true = ctx.y_true.detach().cpu().numpy()[:, None]
                        y_pred = np.argmax(ctx.y_prob.detach().cpu().numpy(), axis=-1)[:, None]
                        total_y_true = np.concatenate(ctx.get('{}_y_true'.format(ctx.cur_data_split)))[:, None]
                        total_y_pred = np.argmax(np.concatenate(
                            ctx.get('{}_y_prob'.format(ctx.cur_data_split))), axis=-1)[:, None]

                        logger.info('(Inner) Epoch: [{}/{}][{}/{}]\t'
                                    'LR: {:.2e}\t'
                                    'Acc: {:.4f} ({:.4f})\t'
                                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                                    .format(ctx.cur_epoch_i + 1,
                                            ctx.num_train_epoch,
                                            cur_step,
                                            ctx.cfg.trainer.train_steps,
                                            ctx.cfg.maml.inner_lr,
                                            eval_acc(y_true, y_pred),
                                            eval_acc(total_y_true, total_y_pred),
                                            loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split))))

            if ctx.cur_batch_i + 1 == ctx.num_train_batch:
                if ctx.cfg.federate.load_from:
                    self._test(ctx)
                self._save_model(ctx)

    def _hook_on_batch_end(self, ctx):
        data_batch = ctx.data_batch
        super()._hook_on_batch_end(ctx)
        ctx.data_batch = data_batch

    def _hook_on_fit_end(self, ctx):
        if ctx.cur_data_split == 'train':
            self._hook_on_batch_forward(ctx)
            self._hook_on_batch_forward_regularizer(ctx)

            cur_task = ctx.cfg.data.type
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()
            ctx.optimizer.step()

            if cur_task == 'sts':
                y_true = ctx.y_true.detach().cpu().numpy()
                y_pred = ctx.y_prob.detach().cpu().numpy()
                total_y_true = np.concatenate(ctx.get('{}_y_true'.format(ctx.cur_data_split)))
                total_y_pred = np.concatenate(ctx.get('{}_y_prob'.format(ctx.cur_data_split)))

                if len(y_true) > 1: y_true = y_true.squeeze()
                if len(y_pred) > 1: y_pred = y_pred.squeeze()
                if len(total_y_pred) > 1: total_y_true = total_y_true.squeeze()
                if len(total_y_pred) > 1: total_y_pred = total_y_pred.squeeze()

                logger.info('(Outer) Epoch: [{}/{}][{}/{}]\t'
                            'LR: {:.2e}\t'
                            'Corr: {:.4f} ({:.4f})\t'
                            'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                            .format(ctx.cur_epoch_i + 1,
                                    ctx.num_train_epoch,
                                    ctx.cur_batch_i + 1,
                                    ctx.num_train_batch,
                                    ctx.cfg.optimizer.lr,
                                    compute_sts_metrics(y_pred, y_true)['corr'] if len(y_pred) > 1 else 0,
                                    compute_sts_metrics(total_y_pred, total_y_true)['corr'] if len(total_y_pred) > 1 else 0,
                                    loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split))))

            elif cur_task in {'imdb', 'squad'}:
                y_true = ctx.y_true.detach().cpu().numpy()[:, None]
                y_pred = np.argmax(ctx.y_prob.detach().cpu().numpy(), axis=-1)[:, None]
                total_y_pred = np.argmax(np.concatenate(
                    ctx.get('{}_y_prob'.format(ctx.cur_data_split))), axis=-1)[:, None]
                total_y_true = np.concatenate(ctx.get('{}_y_true'.format(ctx.cur_data_split)))[:, None]

                logger.info('(Outer) Epoch: [{}/{}][{}/{}]\t'
                            'LR: {:.2e}\t'
                            'Acc: {:.4f} ({:.4f})\t'
                            'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                            .format(ctx.cur_epoch_i + 1,
                                    ctx.num_train_epoch,
                                    ctx.cur_batch_i + 1,
                                    ctx.num_train_batch,
                                    ctx.cfg.optimizer.lr,
                                    eval_acc(y_true, y_pred),
                                    eval_acc(total_y_true, total_y_pred),
                                    loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split))))

        ctx.data_batch = None
        ctx.maml = None
        super()._hook_on_fit_end(ctx)


def call_maml_trainer(trainer_type):
    if trainer_type == 'text-dt-maml':
        trainer_builder = MAMLTrainer
        return trainer_builder


register_trainer('text-dt-maml', call_maml_trainer)
