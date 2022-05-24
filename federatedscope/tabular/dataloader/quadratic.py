import numpy as np


def load_quadratic_dataset(config):
    dataset = dict()
    base = np.exp(1 / float(config.federate.client_num - 1) * np.log(12.5 / 0.02))
    for i in range(1, 1+config.federate.client_num):
        a = 0.02 * base**(i-1)
        client_data = dict()
        client_data['train'] = [(a, .0)]
        client_data['val'] = [(a, .0)]
        client_data['test'] = [(a, .0)]
        dataset[i] = client_data
    return dataset, config
