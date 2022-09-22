from federatedscope.core.auxiliaries.enums import MODE
from federatedscope.register import register_trainer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import Context
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.enums import LIFECYCLE
from federatedscope.core.auxiliaries import utils
from torchviz import make_dot, make_dot_from_trace
import torch
import numpy as np
import copy


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
        self.z1, self.z2 = torch.empty(1), torch.empty(1)
        self.num_samples = 0
        self.local_loss_ratio = 0.5
        self.global_loss_ratio = 1 - self.local_loss_ratio

    
    def get_train_pred_embedding(self): 
        model = self.ctx.model.to(self.ctx.device)
        x1, x2 = torch.cat(self.batches_aug_data_1, dim=0).to(self.ctx.device), torch.cat(self.batches_aug_data_2, dim=0).to(self.ctx.device)
        z1, z2 = model(x1, x2)
        self.batches_aug_data_1, self.batches_aug_data_2 = [], []
        self.z1, self.z2 = z1, z2
        self.ctx.model.to(torch.device('cpu'))
        
        return [self.z1, self.z2]
    
    def _hook_on_batch_forward(self, ctx):
        x, label = [utils.move_to(_, ctx.device) for _ in ctx.data_batch]
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

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_task = ctx.loss_task * self.local_loss_ratio
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        
    def _hook_on_batch_end(self, ctx):
        # update statistics
        ctx.num_samples += ctx.batch_size
        if ctx.cur_mode in [MODE.TRAIN]:
            self.num_samples = ctx.num_samples
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_prob.append(ctx.y_prob[0].detach().cpu().numpy())
        
    def train_with_global_loss(self, loss):
        """
        Arguments:
            loss: loss after global calculate
        :returns:
            grads: grads to optimize the model of other clients
        """

        self.ctx.model = self.ctx.model.to(self.ctx.device)

#         self.ctx.optimizer.zero_grad()

        loss = loss.requires_grad_() * self.global_loss_ratio
        loss.backward()
        
            
        self.ctx.optimizer.step()
        
        return self.ctx.model.state_dict()

        
class LPTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(LPTrainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        
        if config.federate.restore_from != '':
            self.load_model(config.federate.restore_from)

    
def call_cl_trainer(trainer_type):
    if trainer_type == 'cltrainer':
        trainer_builder = CLTrainer
        return trainer_builder
    elif trainer_type == 'lptrainer':
        trainer_builder = LPTrainer
        return trainer_builder 


register_trainer('cltrainer', call_cl_trainer)
