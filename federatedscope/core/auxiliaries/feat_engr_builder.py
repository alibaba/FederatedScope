import logging
from importlib import import_module

logger = logging.getLogger(__name__)


def dummy_wrapper(worker):
    return worker


def get_feat_engr_wrapper(config):
    """
    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        wrapper for client and wrapper for server.
    """
    if config.feat_engr.type == '':
        return dummy_wrapper, dummy_wrapper

    logger.info(f'Feature engineering only works on tabular data, please '
                f'check your `data.type` {config.data.type}.')

    wrap_client = \
        getattr(import_module(f'federatedscope.core.feature.'
                              f'{config.feat_engr.scenario}'),
                f'wrap_{config.feat_engr.type}_client')

    wrap_server = \
        getattr(import_module(f'federatedscope.core.feature.'
                              f'{config.feat_engr.scenario}'),
                f'wrap_{config.feat_engr.type}_server')

    return wrap_client, wrap_server
