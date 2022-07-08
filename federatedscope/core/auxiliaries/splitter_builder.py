import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)


def get_splitter(config):
    client_num = config.federate.client_num
    if config.data.splitter_args:
        args = config.data.splitter_args[0]
    else:
        args = {}

    for func in register.splitter_dict.values():
        splitter = func(config)
        if splitter is not None:
            return splitter
    # Delay import
    # generic splitter
    if config.data.splitter == 'lda':
        from federatedscope.core.splitters.generic import LDASplitter
        splitter = LDASplitter(client_num, **args)
    # graph splitter
    elif config.data.splitter == 'louvain':
        from federatedscope.core.splitters.graph import LouvainSplitter
        splitter = LouvainSplitter(client_num, **args)
    elif config.data.splitter == 'random':
        from federatedscope.core.splitters.graph import RandomSplitter
        splitter = RandomSplitter(client_num, **args)
    elif config.data.splitter == 'rel_type':
        from federatedscope.core.splitters.graph import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, **args)
    elif config.data.splitter == 'graph_type':
        from federatedscope.core.splitters.graph import GraphTypeSplitter
        splitter = GraphTypeSplitter(client_num, **args)
    elif config.data.splitter == 'scaffold':
        from federatedscope.core.splitters.graph import ScaffoldSplitter
        splitter = ScaffoldSplitter(client_num, **args)
    elif config.data.splitter == 'scaffold_lda':
        from federatedscope.core.splitters.graph import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, **args)
    elif config.data.splitter == 'rand_chunk':
        from federatedscope.core.splitters.graph import RandChunkSplitter
        splitter = RandChunkSplitter(client_num, **args)
    else:
        logger.warning('Splitter is none or not found.')
        splitter = None
    return splitter
