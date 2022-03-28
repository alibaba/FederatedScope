import numpy as np
import torch


def calc_blocal_dissim(last_model, local_updated_models):
    '''
    Arguments:
        last_model (dict): the state of last round.
        local_updated_models (list): each element is ooxx.
    Returns:
        b_local_dissimilarity (dict): the measurements.
    '''
    #for k, v in last_model.items():
    #    print(k, v)
    #for i, elem in enumerate(local_updated_models):
    #    print(i, elem)
    local_grads = []
    weights = []
    local_gnorms = []
    for tp in local_updated_models:
        weights.append(tp[0])
        grads = dict()
        gnorms = dict()
        for k, v in tp[1].items():
            grad = v - last_model[k]
            grads[k] = grad
            gnorms[k] = torch.sum(grad**2)
        local_grads.append(grads)
        local_gnorms.append(gnorms)
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    avg_gnorms = dict()
    global_grads = dict()
    for i in range(len(local_updated_models)):
        gnorms = local_gnorms[i]
        for k, v in gnorms.items():
            if k not in avg_gnorms:
                avg_gnorms[k] = .0
            avg_gnorms[k] += weights[i] * v
        grads = local_grads[i]
        for k, v in grads.items():
            if k not in global_grads:
                global_grads[k] = torch.zeros_like(v)
            global_grads[k] += weights[i] * v
    b_local_dissimilarity = dict()
    for k in avg_gnorms:
        b_local_dissimilarity[k] = np.sqrt(
            avg_gnorms[k].item() / torch.sum(global_grads[k]**2).item())
    return b_local_dissimilarity
