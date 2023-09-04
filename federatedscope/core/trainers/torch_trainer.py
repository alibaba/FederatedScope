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

from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.trainers.trainer import Trainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.data import ClientData
from federatedscope.core.data.wrap_dataset import WrapDataset
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from federatedscope.core.monitors.monitor import Monitor

logger = logging.getLogger(__name__)


class GeneralTorchTrainer(Trainer):
    def get_model_para(self):
        if self.cfg.federate.process_num > 1 or \
                self.cfg.federate.share_local_model or \
                self.cfg.llm.deepspeed.use:
            return self._param_filter(self.ctx.model.state_dict())
        else:
            return self._param_filter(self.ctx.model.cpu().state_dict())

    def setup_data(self, ctx):
        """
        Initialization data by ``cfg``.
        """
        if isinstance(ctx.data, ClientData):
            ctx.data.setup(ctx.cfg)
        else:
            logger.warning(f'The data type should be `ClientData` to '
                           f'enable new `config`, but got '
                           f'{type(ctx.data)} instead.')

    def parse_data(self, data):
        """Populate "${split}_data", "${split}_loader" and "num_${
        split}_data" for different data splits
        """
        init_dict = dict()
        if isinstance(data, dict):
            for split in data.keys():
                if split not in ['train', 'val', 'test']:
                    continue
                init_dict["{}_data".format(split)] = None
                init_dict["{}_loader".format(split)] = None
                init_dict["num_{}_data".format(split)] = 0
                if data.get(split, None) is not None:
                    if isinstance(data.get(split), Dataset):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split))
                    elif isinstance(data.get(split), DataLoader):
                        init_dict["{}_loader".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split).dataset)
                    elif isinstance(data.get(split), dict):
                        init_dict["{}_data".format(split)] = data.get(split)
                        init_dict["num_{}_data".format(split)] = len(
                            data.get(split)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(split))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        merged_param = merge_param_dict(self.ctx.model.state_dict().copy(),
                                        self._param_filter(model_parameters))
        self.ctx.model.load_state_dict(merged_param, strict=strict)

    def evaluate(self, target_data_split_name="test"):
        with torch.no_grad():
            super(GeneralTorchTrainer, self).evaluate(target_data_split_name)

        return self.ctx.eval_metrics

    def register_default_hooks_train(self):
        self.register_hook_in_train(
            self._hook_on_fit_start_numerical_precision, "on_fit_start")
        self.register_hook_in_train(self._hook_on_data_parallel_init,
                                    "on_fit_start")
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

    def register_default_hooks_ft(self):
        self.register_hook_in_ft(self._hook_on_fit_start_numerical_precision,
                                 "on_fit_start")
        self.register_hook_in_ft(self._hook_on_data_parallel_init,
                                 "on_fit_start")
        self.register_hook_in_ft(self._hook_on_fit_start_init, "on_fit_start")
        self.register_hook_in_ft(self._hook_on_fit_start_calculate_model_size,
                                 "on_fit_start")
        self.register_hook_in_ft(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_in_ft(self._hook_on_batch_start_init,
                                 "on_batch_start")
        self.register_hook_in_ft(self._hook_on_batch_forward,
                                 "on_batch_forward")
        self.register_hook_in_ft(self._hook_on_batch_forward_regularizer,
                                 "on_batch_forward")
        self.register_hook_in_ft(self._hook_on_batch_forward_flop_count,
                                 "on_batch_forward")
        self.register_hook_in_ft(self._hook_on_batch_backward,
                                 "on_batch_backward")
        self.register_hook_in_ft(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_ft(self._hook_on_fit_end, "on_fit_end")

    def register_default_hooks_eval(self):
        # test/val
        self.register_hook_in_eval(self._hook_on_fit_start_numerical_precision,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_data_parallel_init,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_fit_start_init,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_batch_start_init,
                                   "on_batch_start")
        self.register_hook_in_eval(self._hook_on_batch_forward,
                                   "on_batch_forward")
        self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_eval(self._hook_on_fit_end, "on_fit_end")

    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            ctx.model = ctx.model.half()

    def _hook_on_data_parallel_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below,
           further modifications should be made to `ctx.model` other object:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Wrap ``nn.Module` to \
            `nn.DataParallel`
            ==================================  ===========================
        """
        if isinstance(ctx.model, torch.nn.DataParallel):
            return

        if len(ctx.cfg.train.data_para_dids):
            ctx.model = \
                torch.nn.DataParallel(ctx.model,
                                      device_ids=ctx.cfg.train.data_para_dids)

    def _hook_on_fit_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.model``                       Move to ``ctx.device``
            ``ctx.optimizer``                   Initialize by ``ctx.cfg``
            ``ctx.scheduler``                   Initialize by ``ctx.cfg``
            ``ctx.loss_batch_total``            Initialize to 0
            ``ctx.loss_regular_total``          Initialize to 0
            ``ctx.num_samples``                 Initialize to 0
            ``ctx.ys_true``                     Initialize to ``[]``
            ``ctx.ys_prob``                     Initialize to ``[]``
            ==================================  ===========================
        """
        # prepare model and optimizer
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Initialize optimizer here to avoid the reuse of optimizers
            # across different routines
            ctx.optimizer = get_optimizer(ctx.model,
                                          **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(ctx.optimizer,
                                          **ctx.cfg[ctx.cur_mode].scheduler)

        # TODO: the number of batch and epoch is decided by the current mode
        #  and data split, so the number of batch and epoch should be
        #  initialized at the beginning of the routine

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_fit_start_calculate_model_size(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track model size
            ==================================  ===========================
        """
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return
        if ctx.monitor.total_model_size == 0:
            ctx.monitor.track_model_size(ctx.models)

    def _hook_on_epoch_start(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.{ctx.cur_split}_loader``      Initialize DataLoader
            ==================================  ===========================
        """
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
                self.cfg, ctx.cur_split)
            setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_split)),
                            ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_split)).reset()

    def _hook_on_batch_start_init(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.data_batch``                  Initialize batch data
            ==================================  ===========================
        """
        # prepare data batch
        try:
            ctx.data_batch = CtxVar(
                next(ctx.get("{}_loader".format(ctx.cur_split))),
                LIFECYCLE.BATCH)
        except StopIteration:
            raise StopIteration

    def _hook_on_batch_forward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.y_true``                      Move to `ctx.device`
            ``ctx.y_prob``                      Forward propagation get y_prob
            ``ctx.loss_batch``                  Calculate the loss
            ``ctx.batch_size``                  Get the batch_size
            ==================================  ===========================
        """
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
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
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except:
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general "
                    "trainer case: "
                    "1) ctx.data_batch = [x, y]; and"
                    "2) the ctx.model takes only x as input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size

    def _hook_on_batch_forward_regularizer(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.loss_regular``                Calculate the regular loss
            ``ctx.loss_task``                   Sum the ``ctx.loss_regular`` \
            and ``ctx.loss``
            ==================================  ===========================
        """
        ctx.loss_regular = CtxVar(
            self.cfg.regularizer.mu * ctx.regularizer(ctx), LIFECYCLE.BATCH)
        ctx.loss_task = CtxVar(ctx.loss_batch + ctx.loss_regular,
                               LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.optimizer``                   Update by gradient
            ``ctx.loss_task``                   Backward propagation
            ``ctx.scheduler``                   Update by gradient
            ==================================  ===========================
        """
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.num_samples``                 Add ``ctx.batch_size``
            ``ctx.loss_batch_total``            Add batch loss
            ``ctx.loss_regular_total``          Add batch regular loss
            ``ctx.ys_true``                     Append ``ctx.y_true``
            ``ctx.ys_prob``                     Append ``ctx.ys_prob``
            ==================================  ===========================
        """
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())

    def _hook_on_fit_end(self, ctx):
        """
        Evaluate metrics.

        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.ys_true``                     Convert to ``numpy.array``
            ``ctx.ys_prob``                     Convert to ``numpy.array``
            ``ctx.monitor``                     Evaluate the results
            ``ctx.eval_metrics``                Get evaluated results from \
            ``ctx.monitor``
            ==================================  ===========================
        """
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
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

    def discharge_model(self):
        """
        Discharge the model from GPU device
        """
        # Avoid memory leak
        if torch is None:
            return

        if not self.cfg.federate.share_local_model and \
                not self.cfg.llm.deepspeed.use:
            self.ctx.model.to(torch.device("cpu"))
