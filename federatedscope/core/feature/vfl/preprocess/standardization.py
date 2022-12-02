import logging
import numpy as np

from federatedscope.core.feature.utils import merge_splits_feat

logger = logging.getLogger(__name__)


def wrap_standardization(worker):
    """
    This function is to perform z-norm/standardization for vfl tabular data.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker z-norm/standardization data
    """
    logger.info('Start to execute standardization.')

    # Merge train & val & test
    merged_feat, _ = merge_splits_feat(worker.data)

    feat_mean = np.mean(merged_feat, axis=0)
    feat_std = np.std(merged_feat, axis=0)

    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                split_data['x'] = (split_data['x'] - feat_mean) / feat_std
    worker._init_data_related_var()
    return worker


def wrap_standardization_client(worker):
    return wrap_standardization(worker)


def wrap_standardization_server(worker):
    return wrap_standardization(worker)
