import numpy as np
import torch
import torch.nn.functional as F


def GreedyLoss(pred_feats, true_feats, pred_missing, true_missing, num_pred):
    r"""Greedy loss is a loss function of cacluating the MSE loss for the feature.
    https://proceedings.neurips.cc//paper/2021/file/ \
    34adeb8e3242824038aa65460a47c29e-Paper.pdf
    Fedsageplus models from the "Subgraph Federated Learning with Missing
    Neighbor Generation" (FedSage+) paper, in NeurIPS'21
    Source: https://github.com/zkhku/fedsage

    Arguments:
        pred_feats (torch.Tensor): generated missing features
        true_feats (torch.Tensor): real missing features
        pred_missing (torch.Tensor): number of predicted missing node
        true_missing (torch.Tensor): number of missing node
        num_pred (int): hyperparameters which limit the maximum value of the \
        prediction
    :returns:
        loss : the Greedy Loss
    :rtype:
        torch.FloatTensor
    """
    CUDA, device = (pred_feats.device.type != 'cpu'), pred_feats.device
    if CUDA:
        true_missing = true_missing.cpu()
        pred_missing = pred_missing.cpu()
    loss = torch.zeros(pred_feats.shape)
    if CUDA:
        loss = loss.to(device)
    pred_len = len(pred_feats)
    pred_missing_np = np.round(
        pred_missing.detach().numpy()).reshape(-1).astype(np.int32)
    true_missing_np = true_missing.detach().numpy().reshape(-1).astype(
        np.int32)
    true_missing_np = np.clip(true_missing_np, 0, num_pred)
    pred_missing_np = np.clip(pred_missing_np, 0, num_pred)
    for i in range(pred_len):
        for pred_j in range(min(num_pred, pred_missing_np[i])):
            if true_missing_np[i] > 0:
                if isinstance(true_feats[i][true_missing_np[i] - 1],
                              np.ndarray):
                    true_feats_tensor = torch.tensor(
                        true_feats[i][true_missing_np[i] - 1])
                    if CUDA:
                        true_feats_tensor = true_feats_tensor.to(device)
                else:
                    true_feats_tensor = true_feats[i][true_missing_np[i] - 1]
                loss[i][pred_j] += F.mse_loss(
                    pred_feats[i][pred_j].unsqueeze(0).float(),
                    true_feats_tensor.unsqueeze(0).float()).squeeze(0)

                for true_k in range(min(num_pred, true_missing_np[i])):
                    if isinstance(true_feats[i][true_k], np.ndarray):
                        true_feats_tensor = torch.tensor(true_feats[i][true_k])
                        if CUDA:
                            true_feats_tensor = true_feats_tensor.to(device)
                    else:
                        true_feats_tensor = true_feats[i][true_k]

                    loss_ijk = F.mse_loss(
                        pred_feats[i][pred_j].unsqueeze(0).float(),
                        true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                    if torch.sum(loss_ijk) < torch.sum(loss[i][pred_j].data):
                        loss[i][pred_j] = loss_ijk
            else:
                continue
    return loss.unsqueeze(0).mean().float()
