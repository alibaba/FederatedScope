'''
This implementation is adapted from https://github.com/OpenLMLab/LOMO
(MIT License)

Copyright (c) 2023 OpenLMLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

try:
    import torch
    from torch.optim import Optimizer
except ImportError:
    torch = None

from federatedscope.register import register_optimizer


class LOMO(Optimizer):
    """
    an optimizer for LOMOTrainer
    """
    def __init__(self,
                 model,
                 lr=1e-3,
                 clip_grad_norm=None,
                 clip_grad_value=None):
        if torch is None:
            raise ImportError(
                'Currently, LOMO optimizer is only implemented with `pytorch`')
        self.model = model
        self.lr = lr
        self.local_rank = 0
        self.world_size = 1
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm should be positive,"
                             f" got {self.clip_grad_norm}.")
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        # check if zero3 is enabled
        # please refer the following url for this checking clause
        # https://deepspeed.readthedocs.io/en/stable
        # /_modules/deepspeed/runtime/zero/stage3.html
        p0 = list(self.model.parameters())[0]
        if hasattr(p0, 'ds_tensor'):  # zero3 is enabled
            self.grad_func = self.fuse_update_zero3()
        else:
            self.grad_func = self.fuse_update()
        # TODO temporarily removed for test lomo
        self.loss_scaler = None
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(lr=lr,
                        clip_grad_norm=clip_grad_norm,
                        clip_grad_value=clip_grad_value)
        super(LOMO, self).__init__(self.model.parameters(), defaults)

    def fuse_update(self):
        """
        update model parameters in non-ZeRO mode

        :return: a closure function used for updating model parameters
        """
        def func(x):
            with torch.no_grad():
                for _, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        # if there exists `loss scaler`
                        if self.loss_scaler is not None:
                            # if `overflow has occured` or
                            #   `overflow is occuring`
                            if self.loss_scaler.has_overflow_serial or \
                                    self.loss_scaler._has_inf_or_nan(p.grad):
                                # some params have encoutered with overflow,
                                # drop the gradient, stop the loop
                                p.grad = None
                                self.loss_scaler.has_overflow_serial = True
                                break
                        grad_fp32 = p.grad.to(torch.float32)
                        # clear the calculated gradient for memory consumption
                        p.grad = None
                        if self.loss_scaler is not None:
                            grad_fp32.div_(self.loss_scaler.loss_scale)
                        if self.gather_norm:
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            if self.clip_grad_value is not None \
                                    and self.clip_grad_value > 0:
                                # Clipping gradients by their value
                                grad_fp32.clamp_(min=-self.clip_grad_value,
                                                 max=self.clip_grad_value)
                            if self.clip_grad_norm is not None \
                                    and self.clip_grad_norm > 0 \
                                    and self.clip_coef is not None:
                                # Normalize the gradient according to its norm
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
                for _, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(
                            p.grad,
                            op=torch.distributed.ReduceOp.AVG,
                            async_op=False)
                        # if there exists `loss scaler`
                        if self.loss_scaler is not None:
                            # if `overflow has occured` or
                            #   `overflow is occuring`
                            if self.loss_scaler.has_overflow_serial or \
                                    self.loss_scaler._has_inf_or_nan(p.grad):
                                # some params have encoutered with overflow,
                                # drop the gradient, stop the loop
                                p.grad = None
                                self.loss_scaler.has_overflow_serial = True
                                break
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        param_fp32 = p.ds_tensor.to(torch.float32)
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)

                        if self.gather_norm:
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:  # update param
                            one_dim_grad_fp32 = grad_fp32.view(-1)
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.local_rank
                            end = min(start + partition_size,
                                      grad_fp32.numel())
                            partitioned_grad_fp32 = one_dim_grad_fp32.narrow(
                                0, start, end - start)

                            if self.clip_grad_value is not None:
                                # Clipping gradients by their value
                                partitioned_grad_fp32.clamp_(
                                    min=-self.clip_grad_value,
                                    max=self.clip_grad_value)
                            if self.clip_grad_norm is not None \
                                    and self.clip_grad_norm > 0 \
                                    and self.clip_coef is not None:
                                # Normalize the gradient according to its norm
                                partitioned_grad_fp32.mul_(self.clip_coef)

                            partitioned_p = param_fp32.narrow(
                                0, 0, end - start)
                            partitioned_p.add_(partitioned_grad_fp32,
                                               alpha=-self.lr)
                            p.ds_tensor[:end - start] = partitioned_p
            return x

        return func

    def fused_backward(self, loss, lr):
        """
        update the model parameters based on
        the gradient by a step of backpropagation

        :param loss: loss value, scalar
        :param lr: learning rate
        """
        self.lr = lr
        # Users need call grad_norm themselves and then call backward_step
        if self.clip_grad_norm is not None \
                and self.clip_grad_norm > 0 and self.clip_coef is None:
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm()"
                " before optimizer.fused_backward().")
        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        loss.backward()
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
        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            with torch.no_grad():  # clear gradients
                for n, p in self.model.named_parameters():
                    p.grad = None
            return

        with torch.no_grad():
            # The norm is computed over all gradients together,
            # as if they were concatenated into a single vector.
            # Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            if self.clip_grad_norm is not None:
                self.clip_coef = float(
                    self.clip_grad_norm) / (total_norm + 1e-6)
                self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False


def call_lomo_optimizer(model, type, lr, **kwargs):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        optimizer = None

    if type == 'lomo':
        if optim is not None:
            optimizer = LOMO(model,
                             lr=lr,
                             clip_grad_norm=None,
                             clip_grad_value=None)
        return optimizer


register_optimizer('lomo', call_lomo_optimizer)
