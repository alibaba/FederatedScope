import torch
import torch.nn as nn
import torch.nn.functional as F

from federatedscope.register import register_criterion


class NT_xentloss(nn.Module):
    r"""
    NT_xentloss definition adapted from https://github.com/PatrickHua/SimSiam
    Arguments:
        z1 (torch.tensor): the embedding of model .
        z2 (torch.tensor): the embedding of model using another augmentation.
    returns:
        loss: the NT_xentloss loss for this batch data
    :rtype:
        torch.FloatTensor
    """
    def __init__(self, temperature=0.1):
        super(NT_xentloss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=-1)

        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * N, device=device,
                             dtype=torch.int64)  # scalar label per sample
        loss = F.cross_entropy(logits, labels, reduction='sum') / (2 * N)

        return loss


def create_NT_xentloss(type, device):

    if type == 'NT_xentloss':
        criterion = NT_xentloss().to(device)

        return criterion


register_criterion('NT_xentloss', create_NT_xentloss)
