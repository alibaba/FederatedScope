from federatedscope.register import register_criterion


def call_my_criterion(type, device):
    try:
        import torch.nn as nn
    except ImportError:
        nn = None
        criterion = None

    if type == 'mycriterion':
        if nn is not None:
            criterion = nn.CrossEntropyLoss().to(device)
        return criterion


register_criterion('mycriterion', call_my_criterion)
