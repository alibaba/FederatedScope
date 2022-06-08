try:
    import torch
except ImportError:
    torch = None


def get_optimizer(model, type, lr, **kwargs):
    if torch is None:
        return None
    if isinstance(type, str):
        if hasattr(torch.optim, type):
            if isinstance(model, torch.nn.Module):
                return getattr(torch.optim, type)(model.parameters(), lr,
                                                  **kwargs)
            else:
                return getattr(torch.optim, type)(model, lr, **kwargs)
        else:
            raise NotImplementedError(
                'Optimizer {} not implement'.format(type))
    else:
        raise TypeError()
