import logging

from federatedscope.core.sampler import UniformSampler, GroupSampler

logger = logging.getLogger(__name__)


def get_sampler(sample_strategy,
                client_num=None,
                client_info=None,
                bins=10):
    if sample_strategy == 'uniform':
        return UniformSampler()
    elif sample_strategy == 'group':
        return GroupSampler(client_info=client_info['client_resource'],
                            bins=bins)
    else:
        return None