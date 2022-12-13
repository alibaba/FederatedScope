import logging
import numpy as np

from federatedscope.core.feature.utils import merge_splits_feat

logger = logging.getLogger(__name__)


def wrap_variance_filter(worker):
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

    # If variance is smaller than threshold, the feature will be removed
    feat_var = np.var(merged_feat, axis=0)
    threshold = worker._cfg.feat_engr.selec_threshold
    filtered_col = (feat_var < threshold).nonzero()[0]

    # Filter feature
    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                split_data['x'] = \
                    np.delete(split_data['x'], filtered_col, axis=1)
    worker._init_data_related_var()
    return worker


def wrap_variance_filter_client(worker):
    return wrap_variance_filter(worker)


def wrap_variance_filter_server(worker):
    return wrap_variance_filter(worker)
