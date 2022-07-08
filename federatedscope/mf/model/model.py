from torch.nn import Parameter
from torch.nn import Module

import numpy as np
import torch


class BasicMFNet(Module):
    """Basic model for MF task

    Arguments:
        num_user (int): the number of users
        num_item (int): the number of items
        num_hidden (int): the dimension of embedding vector
    """
    def __init__(self, num_user, num_item, num_hidden):
        super(BasicMFNet, self).__init__()

        self.embed_user = Parameter(
            torch.normal(mean=0,
                         std=0.1,
                         size=(num_user, num_hidden),
                         requires_grad=True,
                         dtype=torch.float32))
        self.register_parameter('embed_user', self.embed_user)
        self.embed_item = Parameter(
            torch.normal(mean=0,
                         std=0.1,
                         size=(num_item, num_hidden),
                         requires_grad=True,
                         dtype=torch.float32))
        self.register_parameter('embed_item', self.embed_item)

    def forward(self, indices, ratings):
        pred = torch.matmul(self.embed_user, self.embed_item.T)
        label = torch.sparse_coo_tensor(indices,
                                        ratings,
                                        size=pred.shape,
                                        device=pred.device,
                                        dtype=torch.float32).to_dense()
        mask = torch.sparse_coo_tensor(indices,
                                       np.ones(len(ratings)),
                                       size=pred.shape,
                                       device=pred.device,
                                       dtype=torch.float32).to_dense()

        return mask * pred, label, float(np.prod(pred.size())) / len(ratings)

    def load_state_dict(self, state_dict, strict: bool = True):

        state_dict[self.name_reserve] = getattr(self, self.name_reserve)
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # Mask embed_item
        del state_dict[self.name_reserve]
        return state_dict


class VMFNet(BasicMFNet):
    """MF model for vertical federated learning

    """
    name_reserve = "embed_item"


class HMFNet(BasicMFNet):
    """MF model for horizontal federated learning

    """
    name_reserve = "embed_user"
