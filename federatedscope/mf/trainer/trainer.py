from federatedscope.mf.dataloader.dataloader import MFDataLoader
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxReferVar
from federatedscope.core.trainers.context import CtxStatsVar
from federatedscope.register import register_trainer


class MFTrainer(GeneralTorchTrainer):
    """Trainer for MF task

    Arguments:
        model (torch.nn.module): MF model.
        data (dict): input data
        device (str): device.
    """
    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different modes

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
            f"{ctx.cur_mode}_avg_loss": ctx.mode.loss_batch_total /
            ctx.mode.num_samples,
            f"{ctx.cur_mode}_total": ctx.mode.num_samples
        }
        ctx.eval_metrics = results

    def _hook_on_batch_end(self, ctx):
        # Update statistics
        ctx.mode.num_samples += ctx.batch_size
        ctx.mode.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.mode.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_batch_forward(self, ctx):
        indices, ratings = ctx.data_batch
        pred, label, ratio = ctx.model(indices, ratings)
        ctx.loss_batch = CtxReferVar(
            ctx.criterion(pred, label) * ratio, "batch")

        ctx.batch_size = CtxStatsVar(len(ratings), "batch")


def call_mf_trainer(trainer_type):
    if trainer_type == "mftrainer":
        trainer_builder = MFTrainer
        return trainer_builder


register_trainer("mftrainer", call_mf_trainer)
