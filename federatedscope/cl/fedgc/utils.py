import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


def norm(w):
    return torch.norm(torch.cat([v.flatten() for v in w.values()])).item()


def compute_global_NT_xentloss(z1, z2, others_z2=[], temperature=0.5):
    """ computes global NT_xentloss"""
    N, Z = z1.shape 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)

    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

    diag = torch.eye(2*N, dtype=torch.bool)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)

    
    if len(others_z2) != 0:
        for z2_ in others_z2:
            N2, Z2 = z2_.shape
            representations = torch.cat([z1, z2_], dim=0)
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
            mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
            mask[N:,:N] = True
            mask[:N,N:] = True
            negatives_other = similarity_matrix[mask].view(2*N, -1)
            negatives = torch.cat([negatives, negatives_other], dim=1)
            
    
    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(2*N, dtype=torch.int64) # scalar label per sample
    loss = F.cross_entropy(logits, labels, reduction='sum')

    return loss / (2 * N)
