from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.auxiliaries import utils
import numpy as np


class CLTrainer(GeneralTorchTrainer):
    def _hook_on_batch_forward(self, ctx):
        x, label = [utils.move_to(_, ctx.device) for _ in ctx.data_batch]
#         print(len(x), x[0].size(), x[1].size(), label.size())
        x1, x2 = x[0], x[1]
        z1, z2 = ctx.model(x1, x2)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
            
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(z1, z2, LIFECYCLE.BATCH)
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
