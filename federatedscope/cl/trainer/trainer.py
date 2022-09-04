from federatedscope.core.auxiliaries.enums import MODE
from federatedscope.register import register_trainer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.auxiliaries import utils
import torch
import numpy as np


class CLTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(CLTrainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        self.batches_aug_data_1, self.batches_aug_data_2 = [], []

    @torch.no_grad()
    def get_train_pred_embedding(self):
        model = self.ctx.model.to(self.ctx.device)
        ys_prob_1, ys_prob_2 = [], []
        x1, x2 = torch.cat(self.batches_aug_data_1, dim=0), torch.cat(self.batches_aug_data_2, dim=0)
        z1, z2 = model(x1.to(self.ctx.device), x2.to(self.ctx.device))
        ys_prob_1 = z1.detach().cpu()
        ys_prob_2 = z2.detach().cpu()
        self.batches_aug_data_1, self.batches_aug_data_2 = [], []
        
        return [ys_prob_1, ys_prob_2]
    
    def _hook_on_batch_forward(self, ctx):
        x, label = [utils.move_to(_, ctx.device) for _ in ctx.data_batch]
#         print(len(x), x[0].size(), x[1].size(), label.size())
        x1, x2 = x[0], x[1]
        if ctx.cur_mode in [MODE.TRAIN]:
            self.batches_aug_data_1.append(x1)
            self.batches_aug_data_2.append(x2)
        z1, z2 = ctx.model(x1, x2)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
            
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar((z1, z2), LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(z1, z2), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
        
        
    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_prob.append(ctx.y_prob[0].detach().cpu().numpy())
        
    def train_with_global_loss(self, model_para, loss):
        """
        Arguments:
            model_para: model parameters
            loss: loss after global calculate
        :returns:
            grads: grads to optimize the model of other clients
        """

        for key in model_para.keys():
            if isinstance(model_para[key], list):
                model_para[key] = torch.FloatTensor(model_para[key])
        self.ctx.model.load_state_dict(model_para)
        self.ctx.model = self.ctx.model.to(self.ctx.device)

        self.ctx.optimizer.zero_grad()

        loss = loss.requires_grad_()
        loss.backward()
        self.ctx.optimizer.step()
        
        return self.ctx.model.state_dict()

        
class LPTrainer(GeneralTorchTrainer):
    pass
    
def call_cl_trainer(trainer_type):
    if trainer_type == 'cltrainer':
        trainer_builder = CLTrainer
        return trainer_builder
    elif trainer_type == 'lptrainer':
        trainer_builder = LPTrainer
        return trainer_builder 


register_trainer('cltrainer', call_cl_trainer)
