import logging
import copy
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator

logger = logging.getLogger(__name__)


class NormboundingAggregator(ClientsAvgAggregator):
    """
    The server clips each update to reduce the negative impact \
        of malicious updates.
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(NormboundingAggregator, self).__init__(model, device, config)
        self.norm_bound = config.aggregator.BFT_args.normbounding_norm_bound

    def aggregate(self, agg_info):
        """
        To preform aggregation with normbounding aggregation rule
        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        avg_model = self._aggre_with_normbounding(models)
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model

    def _aggre_with_normbounding(self, models):
        models_temp = []
        for each_model in models:
            param = self._flatten_updates(each_model[1])
            if torch.norm(param, p=2) > self.norm_bound:
                scaling_rate = self.norm_bound / torch.norm(param, p=2)
                scaled_param = scaling_rate * param
                models_temp.append(
                    (each_model[0], self._reconstruct_updates(scaled_param)))
            else:
                models_temp.append(each_model)
        return self._para_weighted_avg(models_temp)

    def _flatten_updates(self, model):
        model_update = []
        init_model = self.model.state_dict()
        for key in init_model:
            model_update.append(model[key].view(-1))
        return torch.cat(model_update, dim=0)

    def _reconstruct_updates(self, flatten_updates):
        start_idx = 0
        init_model = self.model.state_dict()
        reconstructed_model = copy.deepcopy(init_model)
        for key in init_model:
            reconstructed_model[key] = flatten_updates[
                start_idx:start_idx + len(init_model[key].view(-1))].reshape(
                    init_model[key].shape)
            start_idx = start_idx + len(init_model[key].view(-1))
        return reconstructed_model
