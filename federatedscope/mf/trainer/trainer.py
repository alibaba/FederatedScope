import numpy
from wandb.wandb_torch import torch

from federatedscope.core.monitors import Monitor
from federatedscope.mf.dataloader.dataloader import MFDataLoader
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.register import register_trainer

import logging

logger = logging.getLogger(__name__)


class MFTrainer(GeneralTorchTrainer):
    """Trainer for MF task

    Arguments:
        model (torch.nn.module): MF model.
        data (dict): input data
        device (str): device.
    """
    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different
        modes

        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode), MFDataLoader):
                        init_dict["{}_loader".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = data.get(
                            mode).n_rating
                    else:
                        raise TypeError(
                            "Type {} is not supported for MFTrainer.".format(
                                type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def _hook_on_fit_end(self, ctx):
        results = {
            f"{ctx.cur_mode}_avg_loss": ctx.get("loss_batch_total_{}".format(
                ctx.cur_mode)) /
            ctx.get("num_samples_{}".format(ctx.cur_mode)),
            f"{ctx.cur_mode}_total": ctx.get("num_samples_{}".format(
                ctx.cur_mode))
        }
        setattr(ctx, 'eval_metrics', results)

    def _hook_on_batch_end(self, ctx):
        # update statistics
        setattr(
            ctx, "loss_batch_total_{}".format(ctx.cur_mode),
            ctx.get("loss_batch_total_{}".format(ctx.cur_mode)) +
            ctx.loss_batch.item() * ctx.batch_size)

        if ctx.get("loss_regular", None) is None or ctx.loss_regular == 0:
            loss_regular = 0.
        else:
            loss_regular = ctx.loss_regular.item()
        setattr(
            ctx, "loss_regular_total_{}".format(ctx.cur_mode),
            ctx.get("loss_regular_total_{}".format(ctx.cur_mode)) +
            loss_regular)
        setattr(
            ctx, "num_samples_{}".format(ctx.cur_mode),
            ctx.get("num_samples_{}".format(ctx.cur_mode)) + ctx.batch_size)

        # clean temp ctx
        ctx.data_batch = None
        ctx.batch_size = None
        ctx.loss_task = None
        ctx.loss_batch = None
        ctx.loss_regular = None
        ctx.y_true = None
        ctx.y_prob = None

    def _hook_on_batch_forward(self, ctx):
        indices, ratings = ctx.data_batch
        pred, label, ratio = ctx.model(indices, ratings)
        ctx.loss_batch = ctx.criterion(pred, label) * ratio

        ctx.batch_size = len(ratings)

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by "
                f"initializing trainer subclasses without passing a valid "
                f"monitor instance."
                f"Plz check whether this is you want.")
            return

        if self.cfg.eval.count_flops and self.ctx.monitor.flops_per_sample \
                == 0:
            # calculate the flops_per_sample
            try:
                indices, ratings = ctx.data_batch
                if isinstance(indices, numpy.ndarray):
                    indices = torch.from_numpy(indices)
                if isinstance(ratings, numpy.ndarray):
                    ratings = torch.from_numpy(ratings)
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(
                    ctx.model, (indices, ratings)).total()
                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied by internal model "
                        "nums as self.mirrored_models=True."
                        "if this is not the case you want, "
                        "please customize the count hook")
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except:
                logger.warning(
                    "current flop count implementation is for general "
                    "NodeFullBatchTrainer case: "
                    "1) the ctx.model takes tuple (indices, ratings) as input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                self.ctx.monitor.flops_per_sample = -1  # warning at the
                # first failure

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_mf_trainer(trainer_type):
    if trainer_type == "mftrainer":
        trainer_builder = MFTrainer
        return trainer_builder


register_trainer("mftrainer", call_mf_trainer)
