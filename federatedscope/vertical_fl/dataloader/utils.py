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
    :rtype: int, ndarray, ndarray
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


class VerticalDataSampler(object):
    """
    VerticalDataSampler is used to sample a subset from data

    Arguments:
        data(dict): data
        replace (bool): Whether the sample is with or without replacement
    """
    def __init__(self,
                 data,
                 replace=False,
                 use_full_trainset=True,
                 feature_frac=1.0):
        assert 'x' in data
        self.data_x = data['x']
        self.data_y = data['y'] if 'y' in data else None
        self.data_size = self.data_x.shape[0]
        self.feature_size = self.data_x.shape[1]
        self.replace = replace
        self.use_full_trainset = use_full_trainset
        self.selected_feature_num = max(1,
                                        int(self.feature_size * feature_frac))
        self.selected_feature_index = None

    def sample_data(self, sample_size, index=None):

        # use the entire dataset
        if self.use_full_trainset:
            return range(len(self.data_x)), self.data_x, self.data_y

        if index is not None:
            sampled_x = self.data_x[index]
            sampled_y = self.data_y[index] if self.data_y is not None else None
        else:
            sample_size = min(sample_size, self.data_size)
            index = np.random.choice(a=self.data_size,
                                     size=sample_size,
                                     replace=self.replace)
            sampled_x = self.data_x[index]
            sampled_y = self.data_y[index] if self.data_y is not None else None

        return index, sampled_x, sampled_y

    def sample_feature(self, x):
        if self.selected_feature_num == self.feature_size:
            return range(x.shape[-1]), x
        else:
            feature_index = np.random.choice(a=self.feature_size,
                                             size=self.selected_feature_num,
                                             replace=False)
            self.selected_feature_index = feature_index

        return feature_index, x[:, feature_index]
