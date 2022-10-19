import logging

from importlib import import_module
from federatedscope.core.data.utils import RegexInverseMap, load_dataset, \
    convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed

import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.data import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.data`, some modules are not '
        f'available.')

# TODO: Add PyGNodeDataTranslator and PyGLinkDataTranslator
# TODO: move splitter to PyGNodeDataTranslator and PyGLinkDataTranslator
TRANS_DATA_MAP = {
    'BaseDataTranslator': [
        '.*?@.*?', 'hiv', 'proteins', 'imdb-binary', 'bbbp', 'tox21', 'bace',
        'sider', 'clintox', 'esol', 'freesolv', 'lipo'
    ],
    'DummyDataTranslator': [
        'toy', 'quadratic', 'femnist', 'celeba', 'shakespeare', 'twitter',
        'subreddit', 'synthetic', 'ciao', 'epinions', '.*?vertical_fl_data.*?',
        '.*?movielens.*?', '.*?cikmcup.*?', 'graph_multi_domain.*?', 'cora',
        'citeseer', 'pubmed', 'dblp_conf', 'dblp_org', 'csbm.*?', 'fb15k-237',
        'wn18'
    ],  # Dummy for FL dataset
}
DATA_TRANS_MAP = RegexInverseMap(TRANS_DATA_MAP, None)


def get_data(config, client_cfgs=None):
    """Instantiate the data and update the configuration accordingly if
    necessary.
    Arguments:
        config: a cfg node object.
        client_cfgs: dict of client-specific cfg node object.
    Returns:
        obj: The dataset object.
        cfg.node: The updated configuration.
    """
    # Fix the seed for data generation
    setup_seed(12345)

    for func in register.data_dict.values():
        data_and_config = func(config, client_cfgs)
        if data_and_config is not None:
            return data_and_config

    # Load dataset from source files
    dataset, modified_config = load_dataset(config)

    # Perform translator to non-FL dataset
    translator = getattr(import_module('federatedscope.core.data'),
                         DATA_TRANS_MAP[config.data.type.lower()])(
                             modified_config, client_cfgs)
    data = translator(dataset)

    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    data = convert_data_mode(data, modified_config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, modified_config
