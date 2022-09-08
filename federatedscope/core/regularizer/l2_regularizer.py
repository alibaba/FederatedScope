from federatedscope.register import register_regularizer
from torch.nn import Module

import torch

REGULARIZER_NAME = "l2_regularizer"


class L2Regularizer(Module):
    """Returns the l2 norm of weight

        Arguments:
            p (int): The order of norm.
            tensor_before: The original matrix or vector
            tensor_after: The updated matrix or vector

        Returns:
            Tensor: the norm of the given udpate.
    """
    def __init__(self):
        super(L2Regularizer, self).__init__()

    def forward(self, ctx):
        l2_norm = 0.
        for param in ctx.model.parameters():
            l2_norm += torch.sum(param**2)
        return l2_norm


def call_l2_regularizer(type):
    if type == REGULARIZER_NAME:
        regularizer = L2Regularizer
        return regularizer


register_regularizer(REGULARIZER_NAME, call_l2_regularizer)
