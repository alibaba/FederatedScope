import numpy as np
import math


def batch_iter(data, batch_size, shuffled=True):
    """
    A batch iteration

    Arguments:
        data(dict): data
        batch_size (int): the batch size
        shuffled (bool): whether to shuffle the data at the start of each epoch
    :returns: sample index, batch of x, batch_of y
    :rtype: int, ndarray, ndarry
    """

    assert 'x' in data and 'y' in data
    data_x = data['x']
    data_y = data['y']
    data_size = len(data_y)
    num_batches_per_epoch = math.ceil(data_size / batch_size)

    while True:
        shuffled_index = np.random.permutation(
            np.arange(data_size)) if shuffled else np.arange(data_size)
        for batch in range(num_batches_per_epoch):
            start_index = batch * batch_size
            end_index = min(data_size, (batch + 1) * batch_size)
            sample_index = shuffled_index[start_index:end_index]
            yield sample_index, data_x[sample_index], data_y[sample_index]
