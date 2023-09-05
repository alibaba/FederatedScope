# This implementation is adapted from https://github.com/OpenLMLab/LOMO (MIT License)

# Copyright (c) 2023 OpenLMLab

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

try:
    import torch
except ImportError:
    torch=None
    raise ImportError('Currently, LOMO optimizer is only implemented with `pytorch`')
from torch.optim import Optimizer

from federatedscope.register import register_optimizer


class LOMO(Optimizer):
    """
    an optimizer for LOMOTrainer
    """

    def __init__(self, model, lr=1e-3, clip_grad_norm=None, clip_grad_value=None):
        self.model = model
        self.lr = lr
        self.local_rank = 0
        self.world_size = 1
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(f"clip_grad_norm should be positive, got {self.clip_grad_norm}.")
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        # check if zero3 is enabled
        p0 = list(self.model.parameters())[0]
        if hasattr(p0, 'ds_tensor'):  # zero3 is enabled
            self.grad_func = self.fuse_update_zero3()
        else:
            self.grad_func = self.fuse_update()
        # check if fp16 is enabled
        if p0.dtype == torch.float16:
            # TODO temporarily removed for test lomo with llama
            self.loss_scaler = None
            # self.loss_scaler = DynamicLossScaler(
            #     init_scale=2 ** 16,
            # )  # TODO: add args
            # if self.clip_grad_norm is None:
            #     raise ValueError(
            #         "Loss scaling is recommended to be used with grad norm to get better performance."
            #     )
        else:
            self.loss_scaler = None

        # register hook function, which will be called through the backward process
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(lr=lr, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)
        super(LOMO, self).__init__(self.model.parameters(), defaults)

    def fuse_update(self):
        """
        update model parameters in non-ZeRO mode

        :return: a closure function used for updating model parameters
        """

        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if self.loss_scaler:
                            if self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                                # if the overflow is detected, drop the gradient
                                p.grad = None
                                self.loss_scaler.has_overflow_serial = True
                                break
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None   # clear the calculated gradient for memory consumption
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            if self.clip_grad_value is not None and self.clip_grad_value > 0:
                                # Clipping gradients by their value
                                grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)
                            p_fp32 = p.data.to(torch.float32)
                            p_fp32.add_(grad_fp32, alpha=-self.lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self):
        """
        update model parameters in ZeRO mode

        :return: a closure function used for updating model parameters
        """
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                        if self.loss_scaler:
                            if self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                                # if the overflow is detected, drop the gradient
                                p.grad = None
                                self.loss_scaler.has_overflow_serial = True
                                break

                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        param_fp32 = p.ds_tensor.to(torch.float32)
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)

                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:  # update param
                            one_dim_grad_fp32 = grad_fp32.view(-1)
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.local_rank
                            end = min(start + partition_size, grad_fp32.numel())
                            partitioned_grad_fp32 = one_dim_grad_fp32.narrow(0, start, end - start)

                            if self.clip_grad_value is not None:
                                # Clipping gradients by their value
                                partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                # Normalize the gradient according to its norm (computed in another pass)
                                partitioned_grad_fp32.mul_(self.clip_coef)

                            partitioned_p = param_fp32.narrow(0, 0, end - start)
                            partitioned_p.add_(partitioned_grad_fp32, alpha=-self.lr)
                            p.ds_tensor[ : end - start] = partitioned_p
            return x

        return func

    def fused_backward(self, loss, lr):
        """
        update the model parameters based on the gradient by a step of backpropagation

        :param loss: loss value, scalar
        :param lr: learning rate
        """
        self.lr = lr
        # Users need call grad_norm themselves and then call backward_step
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is None:
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm() before optimizer.fused_backward()."
            )
        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        loss.backward()
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything. 
        self.grad_func(0)

    def grad_norm(self, loss):
        """
        calculate the norm of gradients

        :param loss: loss value, scala
        """
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything. 
        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            with torch.no_grad():  # clear gradients
                for n, p in self.model.named_parameters():
                    p.grad = None
            return


        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            if self.clip_grad_norm is not None:
                self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
                self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False


class DynamicLossScaler:
    def __init__(self,
                 init_scale=2 ** 32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False,
                 raise_error_at_min_scale=True,
                 dtype=torch.half):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dtype = dtype
        self.has_overflow_serial = False

    @property
    def loss_scale(self):
        return self.cur_scale


    def _has_inf_or_nan(self, x):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
                return True
            return False


    def update_scale(self, overflow):
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.")
                else:
                    next_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                    if torch.distributed.get_rank() == 0:
                        overflow_msg = f"[deepspeed] OVERFLOW! Rank {torch.distributed.get_rank()} Skipping step."
                        if self.dtype == torch.half:
                            overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, reducing to {int(next_scale)}"
                        print(overflow_msg)
                    self.cur_scale = next_scale
            else:
                if torch.distributed.get_rank() == 0:
                    overflow_msg = f"[deepspeed] OVERFLOW! Rank {torch.distributed.get_rank()} Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, but hysteresis is {self.cur_hysteresis}. Reducing hysteresis to {self.cur_hysteresis - 1}"
                    print(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                if torch.distributed.get_rank() == 0:
                    hysteresis_msg = f"Consecutive hysteresis is enabled. Restoring hysteresis to {self.delayed_shift}"
                    print(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1



def call_lomo_optimizer(model, type, lr, **kwargs):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        optimizer = None

    if type == 'lomo':
        if optim is not None:
            optimizer = LOMO(model, lr=lr, clip_grad_norm=None, clip_grad_value=None)
        return optimizer


register_optimizer('lomo', call_lomo_optimizer)
