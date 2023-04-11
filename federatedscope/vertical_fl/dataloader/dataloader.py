import numpy as np


def load_vertical_data(config=None, generate=False):
    """
    To generate the synthetic data for vertical FL

    Arguments:
        config: configuration
        generate (bool): whether to generate the synthetic data
    :returns: The synthetic data, the modified config
    :rtype: dict
    """

    if generate:
        # generate toy data for running a vertical FL example
        INSTANCE_NUM = 1000
        TRAIN_SPLIT = 0.9

        total_dims = np.sum(config.vertical.dims)
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

        # For Server #0
        data[0] = dict()
        data[0]['train'] = None
        data[0]['val'] = None
        data[0]['test'] = test_data

        # For Client #1
        data[1] = dict()
        data[1]['train'] = {
            'x': x[:train_num, :config.vertical.dims[0]],
            'y': y[:train_num]
        }
        data[0]['val'] = None
        data[1]['test'] = test_data

        # For Client #2
        data[2] = dict()
        data[2]['train'] = {'x': x[:train_num, config.vertical.dims[0]:]}
        data[0]['val'] = None
        data[2]['test'] = test_data

        return data, config
    else:
        raise ValueError('You must provide the data file')
