from federatedscope.register import regularizer_dict
from federatedscope.core.regularizer.proximal_regularizer import *
try:
    from torch.nn import Module
except ImportError:
    Module = object


def get_regularizer(type):
    if type is None or type == '':
        return DummyRegularizer()

    for func in regularizer_dict.values():
        regularizer = func(type)
        if regularizer is not None:
            return regularizer()

    raise NotImplementedError(
        "Regularizer {} is not implemented.".format(type))


class DummyRegularizer(Module):
    """Dummy regularizer that only returns zero.

    """
    def __init__(self):
        super(DummyRegularizer, self).__init__()

    def forward(self, ctx):
        return 0.
