from federatedscope.core.trainers.trainer import GeneralTrainer
from federatedscope.register import register_trainer


class MFTrainer(GeneralTrainer):
    """
    model (torch.nn.module): MF model.
    data (dict): input data
    device (str): device.
    """
    def _hook_on_fit_end(self, ctx):
        results = {
            "avg_loss": ctx.get("loss_batch_total_{}".format(ctx.cur_mode)) /
            ctx.get("num_samples_{}".format(ctx.cur_mode))
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
        useridx, items, ratings = ctx.data_batch

        # ratio is to correct the number of loss
        pred, label, ratio = ctx.model(useridx, items, ratings)

        ctx.loss_batch = ctx.criterion(pred, label) * ratio
        ctx.y_prob = pred
        ctx.y_true = label

        ctx.batch_size = len(useridx)


def call_mf_trainer(trainer_type):
    if trainer_type == "mftrainer":
        trainer_builder = MFTrainer
        return trainer_builder


register_trainer("mftrainer", call_mf_trainer)
