import numpy as np


def wrap_min_max_scaling(worker):
    """
    This function is to perform min-max scale vfl tabular data.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker with min-max scaled data
    """
    feat_min, feat_max = [], []
    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                feat_min.append(np.min(split_data['x'], axis=0))
                feat_max.append(np.max(split_data['x'], axis=0))
    feat_min = np.min(feat_min, axis=0)
    feat_max = np.min(feat_max, axis=0)

    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                split_data['x'] = (split_data['x'] - feat_min) / (feat_max -
                                                                  feat_min)
    return worker
