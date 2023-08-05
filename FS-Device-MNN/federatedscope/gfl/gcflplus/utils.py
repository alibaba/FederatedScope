import torch
import numpy as np
import networkx as nx
from dtaidistance import dtw
"""
    Utils from: https://github.com/Oxfordblue7/GCFL
"""


def norm(w):
    return torch.norm(torch.cat([v.flatten() for v in w.values()])).item()


def compute_pairwise_distances(seqs, standardize=False):
    """ computes DTW distances for gcfl+"""
    if standardize:
        # standardize to only focus on the trends
        seqs = np.array(seqs)
        seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
        distances = dtw.distance_matrix(seqs)
    else:
        distances = dtw.distance_matrix(seqs)
    return distances


def min_cut(similarity, cluster):
    g = nx.Graph()
    for i in range(len(similarity)):
        for j in range(len(similarity)):
            g.add_edge(i, j, weight=similarity[i][j])
    cut, partition = nx.stoer_wagner(g)
    c1 = np.array([cluster[x] for x in partition[0]])
    c2 = np.array([cluster[x] for x in partition[1]])
    return c1, c2
