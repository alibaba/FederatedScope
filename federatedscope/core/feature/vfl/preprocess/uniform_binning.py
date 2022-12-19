import logging
import numpy as np

from federatedscope.core.feature.utils import merge_splits_feat, vfl_binning

logger = logging.getLogger(__name__)


def wrap_uniform_binning(worker):
    """
    This function is to perform uniform_binning for vfl tabular data.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker with uniform_binning data
    """
    logger.info('Start to execute uniform binning.')

    # Merge train & val & test
    merged_feat, _ = merge_splits_feat(worker.data)

    # Get bin edges
    if merged_feat is not None:
        num_features = merged_feat.shape[1]
        num_bins = [worker._cfg.feat_engr.num_bins] * num_features
        bin_edges = vfl_binning(merged_feat, num_bins, 'uniform')

    # Transform
    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                for i in range(split_data['x'].shape[1]):
                    split_data['x'][:, i] = \
                        np.searchsorted(bin_edges[i][1:-1],
                                        split_data['x'][:, i], side="right")
    worker._init_data_related_var()
    return worker


def wrap_uniform_binning_client(worker):
    return wrap_uniform_binning(worker)


def wrap_uniform_binning_server(worker):
    return wrap_uniform_binning(worker)
