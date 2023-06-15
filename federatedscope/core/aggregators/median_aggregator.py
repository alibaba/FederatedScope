import copy
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
import logging

logger = logging.getLogger(__name__)


class MedianAggregator(ClientsAvgAggregator):
    """
    Implementation of median refers to `Byzantine-robust distributed
    learning: Towards optimal statistical rates`
    [Yin et al., 2018]
    (http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)

    It computes the coordinate-wise median of recieved updates from clients

    The code is adapted from https://github.com/bladesteam/blades
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(MedianAggregator, self).__init__(model, device, config)
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        assert 2 * self.byzantine_node_num + 2 < config.federate.client_num, \
            "it should be satisfied that 2*byzantine_node_num + 2 < client_num"

    def aggregate(self, agg_info):
        """
        To preform aggregation with Median aggregation rule
        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        avg_model = self._aggre_with_median(models)
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model

    def _aggre_with_median(self, models):
        init_model = self.model.state_dict()
        global_update = copy.deepcopy(init_model)
        for key in init_model:
            temp = torch.stack([each_model[1][key] for each_model in models],
                               0)
            temp_pos, _ = torch.median(temp, dim=0)
            temp_neg, _ = torch.median(-temp, dim=0)
            global_update[key] = (temp_pos - temp_neg) / 2
        return global_update
