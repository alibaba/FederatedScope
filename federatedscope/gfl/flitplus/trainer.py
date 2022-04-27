import torch
from copy import deepcopy
import numpy as np
import torch.nn.functional as F

from federatedscope.gfl.flitplus.vat import VATLoss
from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer


class FLITTrainer(GeneralTorchTrainer):
    def register_default_hooks_train(self):
        super(FLITTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FLITTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        ctx.global_model.to(ctx.device)
        predG = ctx.global_model(batch)
        label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss = lossLocalLabel + torch.relu(lossLocalLabel - lossGlobalLabel.detach())
        weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()

        loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7)) + 1e-7) ** self.cfg.flitplus.tmpFed * (
            lossLocalLabel)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred

    def _hook_on_batch_forward_eval(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


class FLITPlusTrainer(FLITTrainer):
    def _hook_on_batch_forward(self, ctx):
        # LDS should be calculated before the forward for cross entropy
        batch = ctx.data_batch.to(ctx.device)
        ctx.global_model.to(ctx.device)
        vat_loss = VATLoss()  # xi, and eps
        lossLocalVAT = vat_loss(deepcopy(ctx.model), batch)
        lossGlobalVAT = vat_loss(deepcopy(ctx.global_model), batch)

        pred = ctx.model(batch)
        predG = ctx.global_model(batch)
        label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss_loss = lossLocalLabel + torch.relu(lossLocalLabel - lossGlobalLabel.detach())
        weightloss_vat = (lossLocalVAT + torch.relu(lossLocalVAT - lossGlobalVAT.detach()))
        weightloss = weightloss_loss + self.cfg.flitplus.lambdavat * weightloss_vat
        weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()

        loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7)) + 1e-7) ** self.cfg.flitplus.tmpFed * (
                lossLocalLabel + lossLocalVAT)  # weightReg: balance lossLocalLabel and lossLocalVAT
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


def record_initialization(ctx):
    """Record the shared global model to cpu

    """
    ctx.global_model = deepcopy(ctx.model)
    ctx.global_model.to(torch.device("cpu"))

def del_initialization(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.global_model = None