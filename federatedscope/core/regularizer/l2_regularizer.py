from federatedscope.register import register_regularizer
from torch.nn import Module

import torch

REGULARIZER_NAME = "l2_regularizer"


class L2Regularizer(Module):
    """Returns the l2 norm of weight

    Returns:
        Tensor: the norm of the given udpate.
    """
    def __init__(self):
        super(L2Regularizer, self).__init__()

    def forward(self, ctx, skip_bn=False):
        l2_norm = 0.
        for name, param in ctx.model.named_parameters():
            if skip_bn and 'bn' in name:
                continue
            l2_norm += torch.sum(param**2)
        return l2_norm


def call_l2_regularizer(type):
    if type == REGULARIZER_NAME:
        regularizer = L2Regularizer
        return regularizer


register_regularizer(REGULARIZER_NAME, call_l2_regularizer)
