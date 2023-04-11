import numpy as np

from torch.utils.data import DataLoader


def load_quadratic_dataset(config):
    dataset = dict()
    d = config.data.quadratic.dim
    base = np.exp(
        np.log(config.data.quadratic.max_curv / config.data.quadratic.min_curv)
        / (config.federate.client_num - 1))
    for i in range(1, 1 + config.federate.client_num):
        # TODO: enable sphere
        a = 0.02 * base**(i - 1) * np.identity(d)
        # TODO: enable non-zero minimizer, i.e., provide a shift
        client_data = dict()
        client_data['train'] = DataLoader([(a.astype(np.float32), .0)])
        client_data['val'] = DataLoader([(a.astype(np.float32), .0)])
        client_data['test'] = DataLoader([(a.astype(np.float32), .0)])
        dataset[i] = client_data
    return dataset, config
