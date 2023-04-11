from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from copy import deepcopy
import numpy as np


class FedAvgAggregator(object):
    def __init__(self, model=None, device='cpu'):
        self.model = model
        self.device = device

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        avg_model = self._para_weighted_avg(models)

        return avg_model

    def _para_weighted_avg(self, models):

        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        for key in avg_model:
            for i in range(len(models)):
                local_sample_size, local_model = models[i]
                weight = local_sample_size / training_set_size
                if i == 0:
                    avg_model[key] = np.asarray(local_model[key]) * weight
                else:
                    avg_model[key] += np.asarray(local_model[key]) * weight

        return avg_model

    def update(self, model_parameters):
        '''
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        self.model.load_state_dict(model_parameters)
