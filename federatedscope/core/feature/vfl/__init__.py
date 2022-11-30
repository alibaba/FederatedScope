from federatedscope.core.feature.vfl.preprocess.instance_norm import \
    wrap_instance_norm_client, wrap_instance_norm_server
from federatedscope.core.feature.vfl.preprocess.min_max_norm import \
    min_max_norm_client, min_max_norm_server

__all__ = [
    'wrap_instance_norm_client', 'wrap_instance_norm_server',
    'min_max_norm_client', 'min_max_norm_server'
]
