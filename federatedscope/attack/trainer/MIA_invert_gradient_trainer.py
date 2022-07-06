import logging
from typing import Type

import torch

from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from federatedscope.attack.auxiliary.MIA_get_target_data import get_target_data

logger = logging.getLogger(__name__)


def wrap_GradientAscentTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    '''
    wrap the gradient_invert trainer

    Args:
        base_trainer: Type: core.trainers.GeneralTorchTrainer

    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer

    '''

    # base_trainer.ctx.target_data = get_target_data()
    base_trainer.ctx.target_data_dataloader = WrapDataset(
        get_target_data(base_trainer.cfg.data.type))
    base_trainer.ctx.target_data = get_target_data(base_trainer.cfg.data.type)

    base_trainer.ctx.is_target_batch = False
    base_trainer.ctx.finish_injected = False

    base_trainer.ctx.target_data_loss = []

    base_trainer.ctx.outdir = base_trainer.cfg.outdir
    base_trainer.ctx.round = -1
    base_trainer.ctx.inject_round = base_trainer.cfg.attack.inject_round

    base_trainer.register_hook_in_train(new_hook=hook_on_fit_start_count_round,
                                        trigger='on_fit_start',
                                        insert_mode=-1)

    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_start_replace_data_batch,
        trigger='on_batch_start',
        insert_mode=-1)

    base_trainer.replace_hook_in_train(
        new_hook=hook_on_batch_backward_invert_gradient,
        target_trigger='on_batch_backward',
        target_hook_name='_hook_on_batch_backward')

    base_trainer.register_hook_in_train(
        new_hook=hook_on_fit_start_loss_on_target_data,
        trigger='on_fit_start',
        insert_mode=-1)

    # plot the target data loss at the end of fitting

    return base_trainer


def hook_on_fit_start_count_round(ctx):
    ctx.round += 1
    logger.info("============== round: {} ====================".format(
        ctx.round))


def hook_on_batch_start_replace_data_batch(ctx):
    # replace the data batch to the target data
    # check whether need to replace the data; if yes, replace the current
    # batch to target batch
    if ctx.finish_injected == False and ctx.round >= ctx.inject_round:
        logger.info("---------- inject the target data ---------")
        ctx["data_batch"] = ctx.target_data
        ctx.is_target_batch = True
        logger.info(ctx.target_data[0].size())
    else:
        ctx.is_target_batch = False


def hook_on_batch_backward_invert_gradient(ctx):
    if ctx.is_target_batch:
        # if the current data batch is the target data, perform gradient ascent
        ctx.optimizer.zero_grad()
        ctx.loss_batch.backward()
        original_grad = []

        for param in ctx["model"].parameters():
            original_grad.append(param.grad.detach())
            param.grad = -1 * param.grad

        modified_grad = []
        for param in ctx.model.parameters():
            modified_grad.append(param.grad.detach())

        ctx["optimizer"].step()
        logger.info('-------------- Gradient ascent finished -------------')
        ctx.finish_injected = True

    else:
        # if current batch is not target data, perform regular backward step
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()


def hook_on_fit_start_loss_on_target_data(ctx):
    if ctx.finish_injected:
        tmp_loss = []
        x, label = [_.to(ctx.device) for _ in ctx.target_data]
        logger.info(x.size())
        num_target = x.size()[0]

        for i in range(num_target):
            x_i = x[i, :].unsqueeze(0)
            label_i = label[i].reshape(-1)
            pred = ctx.model(x_i)
            tmp_loss.append(
                ctx.criterion(pred, label_i).detach().cpu().numpy())
        ctx.target_data_loss.append(tmp_loss)
