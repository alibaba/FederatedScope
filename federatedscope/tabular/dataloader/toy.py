import copy
import pickle

import numpy as np

from federatedscope.core.data.wrap_dataset import WrapDataset


def load_toy_data(config=None):
    def _generate_data(client_num=5,
                       instance_num=1000,
                       feature_num=5,
                       save_data=False):
        """
        Generate data in FedRunner format
        Args:
            client_num:
            instance_num:
            feature_num:
            save_data:

        Returns:
            {
                '{client_id}': {
                    'train': {
                        'x': ...,
                        'y': ...
                    },
                    'test': {
                        'x': ...,
                        'y': ...
                    },
                    'val': {
                        'x': ...,
                        'y': ...
                    }
                }
            }

        """
        weights = np.random.normal(loc=0.0, scale=1.0, size=feature_num)
        bias = np.random.normal(loc=0.0, scale=1.0)
        data = dict()
        for each_client in range(1, client_num + 1):
            data[each_client] = dict()
            client_x = np.random.normal(loc=0.0,
                                        scale=0.5 * each_client,
                                        size=(instance_num, feature_num))
            client_y = np.sum(client_x * weights, axis=-1) + bias
            client_y = np.expand_dims(client_y, -1)
            client_data = {'x': client_x, 'y': client_y}
            data[each_client]['train'] = client_data

        # test data
        test_x = np.random.normal(loc=0.0,
                                  scale=1.0,
                                  size=(instance_num, feature_num))
        test_y = np.sum(test_x * weights, axis=-1) + bias
        test_y = np.expand_dims(test_y, -1)
        test_data = {'x': test_x, 'y': test_y}
        for each_client in range(1, client_num + 1):
            data[each_client]['test'] = copy.deepcopy(test_data)

        # val data
        val_x = np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(instance_num, feature_num))
        val_y = np.sum(val_x * weights, axis=-1) + bias
        val_y = np.expand_dims(val_y, -1)
        val_data = {'x': val_x, 'y': val_y}
        for each_client in range(1, client_num + 1):
            data[each_client]['val'] = val_data

        # server_data
        data[0] = dict()
        # data[0]['train'] = None
        data[0]['val'] = val_data
        data[0]['test'] = test_data

        if save_data:
            # server_data = dict()
            save_client_data = dict()

            for client_idx in range(0, client_num + 1):
                if client_idx == 0:
                    filename = 'data/server_data'
                else:
                    filename = 'data/client_{:d}_data'.format(client_idx)
                with open(filename, 'wb') as f:
                    save_client_data['train'] = {
                        k: v.tolist()
                        for k, v in data[client_idx]['train'].items()
                    }
                    save_client_data['val'] = {
                        k: v.tolist()
                        for k, v in data[client_idx]['val'].items()
                    }
                    save_client_data['test'] = {
                        k: v.tolist()
                        for k, v in data[client_idx]['test'].items()
                    }
                    pickle.dump(save_client_data, f)

        return data

    data = _generate_data(client_num=config.federate.client_num,
                          save_data=config.data.save_data)
    for client_id in data.keys():
        data[client_id] = {
            k: WrapDataset(v)
            for k, v in data[client_id].items()
        } if data[client_id] is not None else None

    return data, config
