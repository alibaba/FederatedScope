from federatedscope.register import regularizer_dict
from federatedscope.core.regularizer.proximal_regularizer import *
try:
    from torch.nn import Module
except ImportError:
    Module = object


def get_regularizer(reg_type):
    """
    This function builds an instance of regularizer to regularize training.

    Args:
        reg_type: type of scheduler, such as see \
            https://pytorch.org/docs/stable/optim.html for details

    Returns:
        An instantiated regularizer.
    """
    if reg_type is None or reg_type == '':
        return DummyRegularizer()

    for func in regularizer_dict.values():
        regularizer = func(reg_type)
        if regularizer is not None:
            return regularizer()

    raise NotImplementedError(
        "Regularizer {} is not implemented.".format(reg_type))


class DummyRegularizer(Module):
    """Dummy regularizer that only returns zero.

    """
    def __init__(self):
        super(DummyRegularizer, self).__init__()

    def forward(self, ctx):
        return 0.
