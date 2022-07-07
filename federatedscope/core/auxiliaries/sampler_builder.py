import logging

from federatedscope.core.sampler import UniformSampler, GroupSampler

logger = logging.getLogger(__name__)


def get_sampler(sample_strategy='uniform',
                client_num=None,
                client_info=None,
                bins=10):
    if sample_strategy == 'uniform':
        return UniformSampler(client_num=client_num)
    elif sample_strategy == 'group':
        return GroupSampler(client_num=client_num,
                            client_info=client_info,
                            bins=bins)
    else:
        raise ValueError(
            f"The sample strategy {sample_strategy} has not been provided.")
