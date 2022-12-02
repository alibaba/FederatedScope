import logging
import numpy as np

logger = logging.getLogger(__name__)


def wrap_log_transform(worker):
    """
    This function is to perform log transform for data;
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap worker with log transformed data.
    """
    logger.info('Start to execute log-transform scaling .')

    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(worker.data, split):
            split_data = getattr(worker.data, split)
            if split_data is not None and 'x' in split_data:
                split_data['x'] = np.log(split_data['x'])
    worker._init_data_related_var()
    return worker


def wrap_log_transform_client(worker):
    return wrap_log_transform(worker)


def wrap_log_transform_server(worker):
    return wrap_log_transform(worker)
