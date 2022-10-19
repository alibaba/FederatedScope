import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)


def get_splitter(config):
    client_num = config.federate.client_num
    if config.data.splitter_args:
        kwargs = config.data.splitter_args[0]
    else:
        kwargs = {}

    for func in register.splitter_dict.values():
        splitter = func(client_num, **kwargs)
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
    else:
        logger.warning(f'Splitter {config.data.splitter} not found or not '
                       f'used.')
        splitter = None
    return splitter
