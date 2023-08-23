# The implementations of MeZO optimizer in federatedscope/llm/trainer/mezo_trainer.py is
# adapted from https://github.com/princeton-nlp/MeZO (MIT License)

# Copyright (c) 2021 Princeton Natural Language Processing

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import logging
from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE


logger = logging.getLogger(__name__)

# 先zo_step计算loss，然后zo_update根据loss来更新参数
class MeZOTrainer(LLMTrainer):
    @staticmethod
    def zo_step(ctx, zo_eps=1e-3):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # determine which parameters to optimize 
        ctx.named_parameters_to_optim = []
        for name, param in ctx.model.named_parameters():
            if param.requires_grad:
                ctx.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        ctx.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        MeZOTrainer.zo_perturb_parameters(
            ctx=ctx,
            scaling_factor=1, 
            zo_eps=zo_eps
        )
        logits1, loss1 = MeZOTrainer.zo_forward(ctx)

        # Second function evaluation
        MeZOTrainer.zo_perturb_parameters(
            ctx=ctx,
            scaling_factor=-2, 
            zo_eps=zo_eps
        )
        _, loss2 = MeZOTrainer.zo_forward(ctx)

        ctx.projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

        # No gradient accumulation support
        # TODO double check here to confirm the implementation of gradient accumulation in FS and MeZO
        # assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        MeZOTrainer.zo_perturb_parameters(
            ctx=ctx,
            scaling_factor=1,
            zo_eps=zo_eps
        )
        return logits1, loss1
    
    
    @staticmethod
    def zo_perturb_parameters(ctx, scaling_factor=1, zo_eps=1e-3):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(ctx.zo_random_seed)
        
        for name, param in ctx.named_parameters_to_optim:
            z = torch.normal(
                mean=0, 
                std=1, 
                size=param.data.size(), 
                device=param.data.device, 
                dtype=param.data.dtype
            )
            param.data = param.data + scaling_factor * z * zo_eps


    @staticmethod
    def zo_forward(ctx):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        ctx.model.eval()
        # TODO support of non-differential functions
        # if self.args.non_diff:
        #     # Non-differentiable objective (may require autoregressive generation)
        #     return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return logits.detach(), loss.detach()
            
            
    def _hook_on_batch_forward(self, ctx):
        logits, loss = MeZOTrainer.zo_step(ctx)
        labels = ctx.data_batch['labels'].to(ctx.device)
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        # ctx.test = '123'
        # input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        # labels = ctx.data_batch['labels'].to(ctx.device)
        # attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        # outputs = ctx.model(input_ids=input_ids,
        #                     labels=labels,
        #                     attention_mask=attention_mask)

        # logits = outputs.logits
        # loss = outputs.loss
        
        
    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return
        torch.manual_seed(ctx.zo_random_seed)     
        for name, param in ctx.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - ctx.cfg.train.optimizer.lr * (ctx.projected_grad * z + ctx.cfg.train.optimizer.weight_decay * param.data)
            else:
                param.data = param.data - ctx.cfg.train.optimizer.lr * (ctx.projected_grad * z)

        if ctx.scheduler is not None:
            ctx.scheduler.step()
    
    
def call_mezo_trainer(trainer_type):
    if trainer_type == 'mezotrainer':
        trainer_builder = MeZOTrainer
        return trainer_builder


register_trainer('mezotrainer', call_mezo_trainer)
