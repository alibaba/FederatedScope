import logging
import numpy as np

from federatedscope.core.feature.utils import merge_splits_feat

logger = logging.getLogger(__name__)


def wrap_min_max_norm(worker):
    """
    This function is to perform min-max scale vfl tabular data.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker with min-max scaled data
    """
    logger.info('Start to execute min-max scaling.')

    # Merge train & val & test
    merged_feat, _ = merge_splits_feat(worker.data)

    feat_min = np.min(merged_feat, axis=0)
    feat_max = np.max(merged_feat, axis=0)

    # If max == min, it will be replaced with 0.0
    for col_i in range(len(feat_min)):
        if feat_min[col_i] == feat_max[col_i]:
            feat_max[col_i] = np.inf

    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                split_data['x'] = \
                    (split_data['x'] - feat_min) / (feat_max - feat_min)
    worker._init_data_related_var()
    return worker


def wrap_min_max_norm_client(worker):
    return wrap_min_max_norm(worker)


def wrap_min_max_norm_server(worker):
    return wrap_min_max_norm(worker)
