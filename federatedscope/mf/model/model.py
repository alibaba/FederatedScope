from torch.nn import Module
from torch.nn import Parameter
import torch

import numpy as np


class MFNet(Module):
    def __init__(self, num_user, num_item, num_hidden):
        super(MFNet, self).__init__()

        # Init parameters
        # Only share user embedding
        self.embed_user = Parameter(
            torch.normal(mean=0,
                         std=0.1,
                         size=(num_user, num_hidden),
                         requires_grad=True))
        self.register_parameter('embed_user', self.embed_user)
        self.embed_item = Parameter(
            torch.normal(mean=0,
                         std=0.1,
                         size=(num_item, num_hidden),
                         requires_grad=True))
        self.register_parameter('embed_item', self.embed_item)

    def forward(self, idx_user, item_sets, rating_sets):
        pred = torch.matmul(self.embed_user, self.embed_item.T)

        label = torch.zeros(size=pred.size(), device=pred.device)
        mask = torch.zeros(size=pred.size(), device=pred.device)
        for id_user, items, ratings in zip(idx_user, item_sets, rating_sets):
            for id_item, rating in zip(items, ratings):
                mask[id_user][id_item] = 1.
                label[id_user][id_item] = rating

        pred_mask = pred * mask
        return pred_mask, label, np.prod(mask.shape) / torch.sum(mask)

    def load_state_dict(self,
                        state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        # Mask embed_item
        state_dict["embed_item"] = self.embed_item
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        # Mask embed_item
        del state_dict["embed_item"]
        return state_dict
