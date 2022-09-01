import numpy as np

from torch.utils.data import DataLoader
from federatedscope.core.interface.base_data import ClientData, \
    StandaloneDataDict


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
        client_data = ClientData(DataLoader,
                                 config,
                                 train=[(a.astype(np.float32), .0)],
                                 val=[(a.astype(np.float32), .0)],
                                 test=[(a.astype(np.float32), .0)])
        dataset[i] = client_data
    return StandaloneDataDict(dataset, config), config
