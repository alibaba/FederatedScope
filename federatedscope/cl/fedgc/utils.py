import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(w):
    return torch.norm(torch.cat([v.flatten() for v in w.values()])).item()


class global_NT_xentloss(nn.Module):
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
    def __init__(self, temperature=0.1, device=torch.device("cpu")):
        super(global_NT_xentloss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z1, z2, others_z2=[]):
        N, Z = z1.shape
        z1, z2 = z1.to(self.device), z2.to(self.device)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=-1)

        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

        diag = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)

        if len(others_z2) != 0:
            for z2_ in others_z2:
                z2_ = z2_.detach().to(self.device)
                N2, Z2 = z2_.shape
                representations = torch.cat([z1, z2_], dim=0)
                similarity_matrix = F.cosine_similarity(
                    representations.unsqueeze(1),
                    representations.unsqueeze(0),
                    dim=-1)
                mask = torch.zeros_like(similarity_matrix,
                                        dtype=torch.bool,
                                        device=self.device)
                mask[N:, :N] = True
                mask[:N, N:] = True
                negatives_other = similarity_matrix[mask].view(2 * N, -1)
                negatives = torch.cat([negatives, negatives_other], dim=1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * N, dtype=torch.int64,
                             device=self.device)  # scalar label per sample
        loss = F.cross_entropy(logits, labels, reduction='sum') / (2 * N)

        return loss
