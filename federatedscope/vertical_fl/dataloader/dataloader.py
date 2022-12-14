import numpy as np

from federatedscope.vertical_fl.dataset.adult import Adult
from federatedscope.vertical_fl.dataset.abalone import Abalone

from federatedscope.vertical_fl.dataset.credit \
    import Credit
from federatedscope.vertical_fl.dataset.blog import Blog


def load_vertical_data(config=None, generate=False):
    """
    To load data for vertical FL

    Arguments:
        config: configuration
        generate (bool): whether to generate the synthetic data
    :returns: The data, the modified config
    :rtype: dict
    """

    splits = config.data.splits
    path = config.data.root
    name = config.data.type.lower()
    # TODO: merge the following later
    if config.vertical.use:
        feature_partition = config.vertical_dims
        algo = 'lr'
    elif config.xgb_base.use:
        feature_partition = config.vertical_dims
        algo = 'xgb'
    else:
        raise ValueError('You must provide the data partition')

    if config.data.args:
        args = config.data.args[0]
    else:
        args = {'normalization': False, 'standardization': False}

    if name == 'adult':
        dataset = Adult(root=path,
                        name=name,
                        num_of_clients=config.federate.client_num,
                        feature_partition=feature_partition,
                        tr_frac=splits[0],
                        download=True,
                        seed=1234,
                        args=args,
                        algo=algo)
        data = dataset.data
        return data, config

    elif name == 'credit':
        dataset = Credit(root=path,
                         name=name,
                         num_of_clients=config.federate.client_num,
                         feature_partition=feature_partition,
                         tr_frac=splits[0],
                         download=True,
                         seed=1234,
                         args=args,
                         algo=algo)
        data = dataset.data
        return data, config
    elif name == 'adult':
        dataset = Adult(root=path,
                        name=name,
                        num_of_clients=config.federate.client_num,
                        feature_partition=feature_partition,
                        tr_frac=splits[0],
                        download=True,
                        seed=1234,
                        args=args,
                        algo=algo)
        data = dataset.data
        return data, config
    elif name == 'abalone':
        dataset = Abalone(root=path,
                          name=name,
                          num_of_clients=config.federate.client_num,
                          feature_partition=feature_partition,
                          tr_frac=splits[0],
                          download=True,
                          seed=1234,
                          args=args,
                          algo=algo)
        data = dataset.data
        return data, config
    elif name == 'blog':
        dataset = Blog(root=path,
                       name=name,
                       num_of_clients=config.federate.client_num,
                       feature_partition=feature_partition,
                       tr_frac=splits[0],
                       download=True,
                       seed=1234,
                       args=args,
                       algo=algo)
        data = dataset.data
        return data, config
    elif generate:
        # generate toy data for running a vertical FL example
        INSTANCE_NUM = 1000
        TRAIN_SPLIT = 0.9

        total_dims = np.sum(config.vertical_dims)
        theta = np.random.uniform(low=-1.0, high=1.0, size=(total_dims, 1))
        x = np.random.choice([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0],
                             size=(INSTANCE_NUM, total_dims))
        y = np.asarray([
            1.0 if x >= 0 else -1.0
            for x in np.reshape(np.matmul(x, theta), -1)
        ])

        train_num = int(TRAIN_SPLIT * INSTANCE_NUM)
        test_data = {'theta': theta, 'x': x[train_num:], 'y': y[train_num:]}
        data = dict()

        # For Server
        data[0] = dict()
        data[0]['train'] = None
        data[0]['val'] = None
        data[0]['test'] = test_data

        # For Client #1
        data[1] = dict()
        data[1]['train'] = {'x': x[:train_num, :config.vertical_dims[0]]}
        data[1]['val'] = None
        data[1]['test'] = test_data

        # For Client #2
        data[2] = dict()
        data[2]['train'] = {
            'x': x[:train_num, config.vertical_dims[0]:],
            'y': y[:train_num]
        }
        data[2]['val'] = None
        data[2]['test'] = test_data

        return data, config
    else:
        raise ValueError('You must provide the data file')
