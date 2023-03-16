import copy
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
import logging

logger = logging.getLogger(__name__)


class FLtrustAggregator(ClientsAvgAggregator):
    """
    Implementation of FLtrust refer to `FLTrust: Byzantine-robust\
    Federated Learning via Trust Bootstrapping`
    [Cao et al., 2020]
    (https://www.ndss-symposium.org/wp-content/uploads/\
        ndss2021_6C-2_24434_paper.pdf)

    It computes the trustcore of each update based on a root
    dataset on the server side.
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(FLtrustAggregator, self).__init__(model, device, config)
        self.alpha = config.aggregator.fltrust.global_learningrate

    def aggregate(self, agg_info):
        """
        To preform aggregation with Fltrust aggregation rule
        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        global_delta = agg_info["global_delta"]
        avg_model = self._aggre_with_fltrust(models, global_delta)
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model

    def _aggre_with_fltrust(self, models, global_delta):
        init_model = self.model.state_dict()
        global_update = copy.deepcopy(init_model)
        total_TS = 0
        TSnorm = []
        global_delta_temp = self._flatten_updates(global_delta)
        for each_model in models:
            tmp_delta = copy.deepcopy(self._flatten_updates(each_model[1]))
            TS = torch.dot(tmp_delta, global_delta_temp) / (
                torch.linalg.norm(tmp_delta) *
                torch.linalg.norm(global_delta_temp))
            if TS < 0:
                TS = 0
            total_TS += TS
            norm = torch.linalg.norm(tmp_delta) / torch.linalg.norm(
                global_delta_temp)
            TSnorm.append(TS * norm)
        delta_weight = {}
        for key in init_model:
            delta_weight[key] = TSnorm[0] * models[0][1][key]

        for i in range(1, len(models)):
            for key in init_model:
                delta_weight[key] += TSnorm[i] * models[i][1][key]

        for key in init_model:
            delta_weight[key] /= total_TS
            global_update[key] = delta_weight[key]

        return global_update

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
