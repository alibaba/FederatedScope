import logging
import numpy as np

from federatedscope.core.feature.utils import merge_splits_feat

logger = logging.getLogger(__name__)


def wrap_uniform_binning(worker):
    """
    This function is to perform uniform_binning for vfl tabular data.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker with uniform_binning data
    """
    logger.info('Start to execute min-max scaling.')

    # Merge train & val & test
    merged_feat = merge_splits_feat(worker.data)

    # for split in ['train_data', 'val_data', 'test_data']:
    #     if hasattr(worker.data, split):
    #         split_data = getattr(worker.data, split)
    #         if split_data is not None and 'x' in split_data:
    #             split_data['x'] = \
    #                 (split_data['x'] - feat_min) / (feat_max - feat_min)
    return worker


def wrap_uniform_binning_client(worker):
    return wrap_uniform_binning(worker)


def wrap_uniform_binning_server(worker):
    return wrap_uniform_binning(worker)
