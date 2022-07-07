import torch
from copy import deepcopy

from federatedscope.gfl.loss.vat import VATLoss
from federatedscope.core.trainers.trainer import GeneralTorchTrainer


class FLITTrainer(GeneralTorchTrainer):
    def register_default_hooks_train(self):
        super(FLITTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization_local,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=record_initialization_global,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_global,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FLITTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization_local,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=record_initialization_global,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_global,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        ctx.global_model.to(ctx.device)
        predG = ctx.global_model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FLIT trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss = lossLocalLabel + torch.relu(lossLocalLabel -
                                                 lossGlobalLabel.detach())
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()
        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (lossLocalLabel)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


class FLITPlusTrainer(FLITTrainer):
    def _hook_on_batch_forward(self, ctx):
        # LDS should be calculated before the forward for cross entropy
        batch = ctx.data_batch.to(ctx.device)
        ctx.global_model.to(ctx.device)
        if ctx.cur_mode == 'test':
            lossLocalVAT, lossGlobalVAT = torch.tensor(0.), torch.tensor(0.)
        else:
            vat_loss = VATLoss()  # xi, and eps
            lossLocalVAT = vat_loss(deepcopy(ctx.model), batch,
                                    deepcopy(ctx.criterion))
            lossGlobalVAT = vat_loss(deepcopy(ctx.global_model), batch,
                                     deepcopy(ctx.criterion))

        pred = ctx.model(batch)
        predG = ctx.global_model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FLITPLUS trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss_loss = lossLocalLabel + torch.relu(lossLocalLabel -
                                                      lossGlobalLabel.detach())
        weightloss_vat = (lossLocalVAT +
                          torch.relu(lossLocalVAT - lossGlobalVAT.detach()))
        weightloss = self.cfg.flitplus.lambdavat * \
            weightloss_vat + weightloss_loss
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()

        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (
                    lossLocalLabel +
                    self.cfg.flitplus.weightReg * lossLocalVAT)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


class FedFocalTrainer(GeneralTorchTrainer):
    def register_default_hooks_train(self):
        super(FedFocalTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization_local,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FedFocalTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization_local,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FLIT trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossLocalLabel = ctx.criterion(pred, label)
        weightloss = lossLocalLabel
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()

        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (lossLocalLabel)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


class FedVATTrainer(GeneralTorchTrainer):
    def register_default_hooks_train(self):
        super(FedVATTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization_local,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FedVATTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization_local,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        if ctx.cur_mode == 'test':
            lossLocalVAT = torch.tensor(0.)
        else:
            vat_loss = VATLoss()  # xi, and eps
            lossLocalVAT = vat_loss(deepcopy(ctx.model), batch,
                                    deepcopy(ctx.criterion))

        pred = ctx.model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FedVAT trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        lossLocalLabel = ctx.criterion(pred, label)
        weightloss = lossLocalLabel + self.cfg.flitplus.lambdavat * \
            lossLocalVAT
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()

        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (
                    lossLocalLabel +
                    self.cfg.flitplus.weightReg * lossLocalVAT)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred


def record_initialization_local(ctx):
    """Record weight denomaitor to cpu

    """
    ctx.weight_denomaitor = None


def del_initialization_local(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.weight_denomaitor = None


def record_initialization_global(ctx):
    """Record the shared global model to cpu

    """
    ctx.global_model = deepcopy(ctx.model)
    ctx.global_model.to(torch.device("cpu"))


def del_initialization_global(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.global_model = None
