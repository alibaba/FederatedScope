import torch
import copy
import numpy as np
import torch.nn.functional as F

from vat import VATLoss
from federatedscope.register import register_trainer
from federatedscope.gfl.trainer.graphtrainer import GraphMiniBatchTrainer


class FLITTrainer(GraphMiniBatchTrainer):
    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        # TODO: 怎么插入，把def当成property？怎么调用？
        predG = ctx.weight_init(batch)
        label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss = lossLocalLabel + torch.relu(lossLocalLabel - lossGlobalLabel.detach())
        weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()

        # TODO: 在config里补充 cfg.flitplus.tmpFed = 0.5 ???
        loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7)) + 1e-7) ** self.cfg.flitplus.tmpFed * (
            lossLocalLabel)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


class FLITPlusTrainer(FLITTrainer):
    def _hook_on_batch_forward(self, ctx):
        # TODO: 在FLITTrainer的基础上增加
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        predG = ctx.weight_init(batch)
        label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)
        # TODO: LDS should be calculated before the forward for cross entropy
        vat_loss = VATLoss()  # xi, and eps
        lossLocalVAT = vat_loss(ctx.model, batch)
        lossGlobalVAT = vat_loss(ctx.weight_init, batch)

        weightloss_loss = lossLocalLabel + torch.relu(lossLocalLabel - lossGlobalLabel.detach())
        weightloss_vat = (lossLocalVAT + torch.relu(lossLocalVAT - lossGlobalVAT.detach()))
        # TODO: 在config里补充 cfg.flitplus.lambdavat = 0.01 ???
        weightloss = weightloss_loss + self.cfg.flitplus.lambdavat * weightloss_vat
        weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()

        # TODO:weightReg可以删除？
        loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7)) + 1e-7) ** self.cfg.flitplus.tmpFed * (
                lossLocalLabel + lossLocalVAT)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


# TODO:这两个函数定义应该放在哪里？
def record_initialization(ctx):
    """Record the initialized weights within local updates

    """
    ctx.weight_init = deepcopy(
        [_.data.detach() for _ in ctx.model.parameters()])

def del_initialization(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.weight_init = None

# TODO:这个应该放在哪？
def call_graph_level_trainer(trainer_type):
    if trainer_type == 'flit_trainer':
        trainer_builder = FLITTrainer
        return trainer_builder
    elif trainer_type == 'flitplus_trainer':
        trainer_builder = FLITPlusTrainer
        return trainer_builder


register_trainer('flit_trainer', call_graph_level_trainer)
register_trainer('flitplus_trainer', call_graph_level_trainer)