import os
import logging

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = None

from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.monitors.monitor import Monitor

logger = logging.getLogger(__name__)


class GeneralTorchTrainer(Trainer):
    def get_model_para(self):
        return self._param_filter(
            self.ctx.model.state_dict() if self.cfg.federate.
            share_local_model else self.ctx.model.cpu().state_dict())

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different modes

        """
        # TODO: more robust for different data
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode), Dataset):
                        init_dict["{}_data".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode))
                    elif isinstance(data.get(mode), DataLoader):
                        init_dict["{}_loader".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode).dataset)
                    elif isinstance(data.get(mode), dict):
                        init_dict["{}_data".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train
        if self.ctx.get(
                f"{target_data_split_name}_data") is None and self.ctx.get(
                    f"{target_data_split_name}_loader") is None:
            raise ValueError(
                f"No {target_data_split_name}_data or"
                f" {target_data_split_name}_loader in the trainer")
        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            before_metric = self.evaluate(target_data_split_name='val')

        self._run_routine("train", hooks_set, target_data_split_name)
        result_metric = self.ctx.eval_metrics

        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            after_metric = self.evaluate(target_data_split_name='val')
            result_metric['val_total'] = before_metric['val_total']
            result_metric['val_avg_loss_before'] = before_metric[
                'val_avg_loss']
            result_metric['val_avg_loss_after'] = after_metric['val_avg_loss']

        return self.ctx.num_samples_train, self.get_model_para(), result_metric

    def update(self, model_parameters):
        '''
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        for key in model_parameters:
            if isinstance(model_parameters[key], list):
                model_parameters[key] = torch.FloatTensor(
                    model_parameters[key])
        self.ctx.model.load_state_dict(self._param_filter(model_parameters),
                                       strict=False)

    def evaluate(self, target_data_split_name="test"):
        with torch.no_grad():
            super(GeneralTorchTrainer, self).evaluate(target_data_split_name)

        return self.ctx.eval_metrics

    # def validate(self, target_data_split_name="val"):
    #    with torch.no_grad():
    #        super(GeneralTorchTrainer, self).evaluate(target_data_split_name)

    #    return self.ctx.eval_metrics

    def finetune(self, target_data_split_name="train", hooks_set=None):

        # freeze the parameters during the fine-tune stage
        require_grad_changed_paras = set()
        if self.cfg.trainer.finetune.freeze_param != "":
            preserved_paras = self._param_filter(
                self.ctx.model.state_dict(),
                self.cfg.trainer.finetune.freeze_param)
            for name, param in self.ctx.model.named_parameters():
                if name not in preserved_paras and param.requires_grad is True:
                    param.requires_grad = False
                    require_grad_changed_paras.add(name)

        # change the optimization configs
        original_lrs = []
        for g in self.ctx.optimizer.param_groups:
            original_lrs.append(g['lr'])
            g['lr'] = self.cfg.trainer.finetune.lr
        original_epoch_num = self.ctx["num_train_epoch"]
        original_batch_num = self.ctx["num_train_batch"]
        self.ctx["num_train_epoch"] = 1
        self.ctx["num_train_batch"] = self.cfg.trainer.finetune.steps

        # do the fine-tuning process
        self.train(target_data_split_name, hooks_set)

        # restore the state before fine-tuning
        if len(require_grad_changed_paras) > 0:
            for name, param in self.ctx.model.named_parameters():
                if name in require_grad_changed_paras:
                    param.requires_grad = True

        for i, g in enumerate(self.ctx.optimizer.param_groups):
            g['lr'] = original_lrs[i]

        self.ctx["num_train_epoch"] = original_epoch_num
        self.ctx["num_train_batch"] = original_batch_num

    def register_default_hooks_train(self):
        self.register_hook_in_train(self._hook_on_fit_start_init,
                                    "on_fit_start")
        self.register_hook_in_train(
            self._hook_on_fit_start_calculate_model_size, "on_fit_start")
        self.register_hook_in_train(self._hook_on_epoch_start,
                                    "on_epoch_start")
        self.register_hook_in_train(self._hook_on_batch_start_init,
                                    "on_batch_start")
        self.register_hook_in_train(self._hook_on_batch_forward,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_forward_regularizer,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_forward_flop_count,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_backward,
                                    "on_batch_backward")
        self.register_hook_in_train(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_train(self._hook_on_fit_end, "on_fit_end")

    def register_default_hooks_eval(self):
        # test/val
        self.register_hook_in_eval(self._hook_on_fit_start_init,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_batch_start_init,
                                   "on_batch_start")
        self.register_hook_in_eval(self._hook_on_batch_forward,
                                   "on_batch_forward")
        self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_eval(self._hook_on_fit_end, "on_fit_end")

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)

        # prepare statistics
        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])

    def _hook_on_fit_start_calculate_model_size(self, ctx):
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return
        if self.ctx.monitor.total_model_size == 0:
            self.ctx.monitor.track_model_size(ctx.models)

    def _hook_on_epoch_start(self, ctx):
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_data_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_data_split))),
                self.cfg)
            setattr(ctx, "{}_loader".format(ctx.cur_data_split),
                    ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_data_split)),
                            ReIterator):
            setattr(
                ctx, "{}_loader".format(ctx.cur_data_split),
                ReIterator(ctx.get("{}_loader".format(ctx.cur_data_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_data_split)).reset()

    def _hook_on_batch_start_init(self, ctx):
        # prepare data batch
        try:
            ctx.data_batch = next(
                ctx.get("{}_loader".format(ctx.cur_data_split)))
        except StopIteration:
            raise StopIteration

    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred

        ctx.batch_size = len(label)

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
            the monitoring hook to calculate the flops during the fl course

            Note: for customized cases that the forward process is not only
            based on ctx.model, please override this function (inheritance
            case) or replace this hook (plug-in case)

        :param ctx:
        :return:
        """
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return

        if self.cfg.eval.count_flops and self.ctx.monitor.flops_per_sample \
                == 0:
            # calculate the flops_per_sample
            try:
                x, y = [_.to(ctx.device) for _ in ctx.data_batch]
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(ctx.model, x).total()
                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied "
                        "by internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, "
                        "please customize the count hook")
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except:
                logger.warning(
                    "current flop count implementation is for general "
                    "trainer case: "
                    "1) ctx.data_batch = [x, y]; and"
                    "2) the ctx.model takes only x as input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                self.ctx.monitor.flops_per_sample = -1  # warning at the
                # first failure

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        self.ctx.monitor.total_flops +=\
            self.ctx.monitor.flops_per_sample * ctx.batch_size

    def _hook_on_batch_forward_regularizer(self, ctx):
        ctx.loss_regular = float(
            self.cfg.regularizer.mu) * ctx.regularizer(ctx)
        ctx.loss_task = ctx.loss_batch + ctx.loss_regular

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

    def _hook_on_batch_end(self, ctx):
        # update statistics
        setattr(
            ctx, "loss_batch_total_{}".format(ctx.cur_data_split),
            ctx.get("loss_batch_total_{}".format(ctx.cur_data_split)) +
            ctx.loss_batch.item() * ctx.batch_size)

        if ctx.get("loss_regular", None) is None or ctx.loss_regular == 0:
            loss_regular = 0.
        else:
            loss_regular = ctx.loss_regular.item()
        setattr(
            ctx, "loss_regular_total_{}".format(ctx.cur_data_split),
            ctx.get("loss_regular_total_{}".format(ctx.cur_data_split)) +
            loss_regular)
        setattr(
            ctx, "num_samples_{}".format(ctx.cur_data_split),
            ctx.get("num_samples_{}".format(ctx.cur_data_split)) +
            ctx.batch_size)

        # cache label for evaluate
        ctx.get("{}_y_true".format(ctx.cur_data_split)).append(
            ctx.y_true.detach().cpu().numpy())

        ctx.get("{}_y_prob".format(ctx.cur_data_split)).append(
            ctx.y_prob.detach().cpu().numpy())

        # clean temp ctx
        ctx.data_batch = None
        ctx.batch_size = None
        ctx.loss_task = None
        ctx.loss_batch = None
        ctx.loss_regular = None
        ctx.y_true = None
        ctx.y_prob = None

    def _hook_on_fit_end(self, ctx):
        """Evaluate metrics.

        """
        setattr(
            ctx, "{}_y_true".format(ctx.cur_data_split),
            np.concatenate(ctx.get("{}_y_true".format(ctx.cur_data_split))))
        setattr(
            ctx, "{}_y_prob".format(ctx.cur_data_split),
            np.concatenate(ctx.get("{}_y_prob".format(ctx.cur_data_split))))
        results = self.metric_calculator.eval(ctx)
        setattr(ctx, 'eval_metrics', results)

    def save_model(self, path, cur_round=-1):
        assert self.ctx.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.ctx.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.ctx.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.ctx.device)
            self.ctx.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))
