import logging

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
    elif config.feat_engr.type == 'vfl_min_max_scaling':
        from federatedscope.core.feature.vfl.scaling.minmax import \
            wrap_min_max_scaling
        return wrap_min_max_scaling, wrap_min_max_scaling
    elif config.feat_engr.type == 'vfl_instance_norm':
        from federatedscope.core.feature.vfl.scaling.instance_norm import \
            wrap_instance_norm_for_client, wrap_instance_norm_for_server
        return wrap_instance_norm_for_client, wrap_instance_norm_for_server
    else:
        logger.warning(f'Feature engineering method {config.feat_engr.type} '
                       f'not provide!')
        return dummy_wrapper, dummy_wrapper
