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
        self.num_user, self.num_item = num_user, num_item
        self.embed_user = torch.nn.Embedding(num_user, num_hidden, sparse=True)
        self.embed_item = torch.nn.Embedding(num_item, num_hidden, sparse=True)

    def forward(self, indices, ratings):
        device = self.embed_user.weight.device

        indices = torch.tensor(np.array(indices)).to(device)

        user_embedding = self.embed_user(indices[0])
        item_embedding = self.embed_item(indices[1])

        pred = (user_embedding * item_embedding).sum(dim=1)

        label = torch.tensor(np.array(ratings)).to(device)

        return pred, label

    def load_state_dict(self, state_dict, strict: bool = True):

        state_dict[self.name_reserve + '.weight'] = getattr(
            getattr(self, self.name_reserve), 'weight')
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # Mask embed_item
        del state_dict[self.name_reserve + '.weight']
        return state_dict


class VMFNet(BasicMFNet):
    """MF model for vertical federated learning

    """
    name_reserve = "embed_item"


class HMFNet(BasicMFNet):
    """MF model for horizontal federated learning

    """
    name_reserve = "embed_user"
