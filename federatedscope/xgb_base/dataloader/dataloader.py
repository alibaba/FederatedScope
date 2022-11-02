import numpy as np
import pandas as pd

from federatedscope.xgb_base.dataset.give_me_some_credit \
    import GiveMeSomeCredit
from federatedscope.xgb_base.dataset.adult import Adult
from federatedscope.xgb_base.dataset.abalone import Abalone
from federatedscope.xgb_base.dataset.blog_feedback import BlogFeedback


def load_xgb_base_data(config=None):
    """
    To generate the synthetic data for vertical FL

    Arguments:
        config: configuration
        generate (bool): whether to generate the synthetic data
    :returns: The synthetic data, the modified config
    :rtype: dict
    """
    splits = config.data.splits
    path = config.data.root
    name = config.data.type.lower()
    dataset = None

    if name == 'givemesomecredit':
        dataset = GiveMeSomeCredit(root=path,
                                   name=name,
                                   num_of_clients=config.federate.client_num,
                                   feature_partition=config.xgb_base.dims,
                                   tr_frac=splits[0],
                                   download=True,
                                   seed=1234)
    elif name == 'adult':
        dataset = Adult(root=path,
                        name=name,
                        num_of_clients=config.federate.client_num,
                        feature_partition=config.xgb_base.dims,
                        tr_frac=splits[0],
                        download=True,
                        seed=1234)
    elif name == 'abalone':
        dataset = Abalone(root=path,
                          name=name,
                          num_of_clients=config.federate.client_num,
                          feature_partition=config.xgb_base.dims,
                          tr_frac=splits[0],
                          download=True,
                          seed=1234)
    elif name == 'blogfeedback':
        dataset = BlogFeedback(root=path,
                               name=name,
                               num_of_clients=config.federate.client_num,
                               feature_partition=config.xgb_base.dims,
                               tr_frac=splits[0],
                               download=True,
                               seed=1234)
    else:
        raise ValueError('You must provide the data file')

    data = dataset.data
    return data, config
