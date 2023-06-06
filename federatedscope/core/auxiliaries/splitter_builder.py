import logging

import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.splitter import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.splitter`, some modules are not '
        f'available.')


def get_splitter(config):
    """
    This function is to build splitter to generate simulated federation \
    datasets from non-FL dataset.

    Args:
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        An instance of splitter (see ``core.splitters`` for details)

    Note:
      The key-value pairs of ``cfg.data.splitter`` and domain:
        ===================  ================================================
        Splitter type        Domain
        ===================  ================================================
        lda	                 Generic
        iid                  Generic
        louvain	             Graph (node-level)
        random	             Graph (node-level)
        rel_type	         Graph (link-level)
        scaffold	         Molecular
        scaffold_lda       	 Molecular
        rand_chunk	         Graph (graph-level)
        ===================  ================================================
    """
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}

    for func in register.splitter_dict.values():
        splitter = func(config.data.splitter, client_num, **kwargs)
        if splitter is not None:
            return splitter
    # Delay import
    # generic splitter
    if config.data.splitter == 'lda':
        from federatedscope.core.splitters.generic import LDASplitter
        splitter = LDASplitter(client_num, **kwargs)
    # graph splitter
    elif config.data.splitter == 'louvain':
        from federatedscope.core.splitters.graph import LouvainSplitter
        splitter = LouvainSplitter(client_num, **kwargs)
    elif config.data.splitter == 'random':
        from federatedscope.core.splitters.graph import RandomSplitter
        splitter = RandomSplitter(client_num, **kwargs)
    elif config.data.splitter == 'rel_type':
        from federatedscope.core.splitters.graph import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, **kwargs)
    elif config.data.splitter == 'scaffold':
        from federatedscope.core.splitters.graph import ScaffoldSplitter
        splitter = ScaffoldSplitter(client_num, **kwargs)
    elif config.data.splitter == 'scaffold_lda':
        from federatedscope.core.splitters.graph import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, **kwargs)
    elif config.data.splitter == 'rand_chunk':
        from federatedscope.core.splitters.graph import RandChunkSplitter
        splitter = RandChunkSplitter(client_num, **kwargs)
    elif config.data.splitter == 'iid':
        from federatedscope.core.splitters.generic import IIDSplitter
        splitter = IIDSplitter(client_num)
    elif config.data.splitter == 'meta':
        from federatedscope.core.splitters.generic import MetaSplitter
        splitter = MetaSplitter(client_num)
    else:
        logger.warning(f'Splitter {config.data.splitter} not found or not '
                       f'used.')
        splitter = None
    return splitter
