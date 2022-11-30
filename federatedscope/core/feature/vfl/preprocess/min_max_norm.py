import numpy as np
import logging

logger = logging.getLogger(__name__)


def min_max_norm(worker):
    """
    This function is to perform min-max scale vfl tabular data.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker with min-max scaled data
    """
    # No communicate needed
    logger.info('Start to execute min-max scaling .')
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
                split_data['x'] = \
                    (split_data['x'] - feat_min) / (feat_max - feat_min)
    return worker


def min_max_norm_client(worker):
    return min_max_norm(worker)


def min_max_norm_server(worker):
    return min_max_norm(worker)
