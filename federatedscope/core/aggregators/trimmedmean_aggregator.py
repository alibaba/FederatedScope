import copy
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
import logging

logger = logging.getLogger(__name__)


class TrimmedmeanAggregator(ClientsAvgAggregator):
    """
    Implementation of median refer to `Byzantine-robust distributed
    learning: Towards optimal statistical rates`
    [Yin et al., 2018]
    (http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)

    The code is adapted from https://github.com/bladesteam/blades
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(TrimmedmeanAggregator, self).__init__(model, device, config)
        self.excluded_ratio = \
            config.aggregator.BFT_args.trimmedmean_excluded_ratio
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        assert 2 * self.byzantine_node_num + 2 < config.federate.client_num, \
            "it should be satisfied that 2*byzantine_node_num + 2 < client_num"
        assert self.excluded_ratio < 0.5

    def aggregate(self, agg_info):
        """
        To preform aggregation with trimmedmean aggregation rule
        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        avg_model = self._aggre_with_trimmedmean(models)
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model

    def _aggre_with_trimmedmean(self, models):
        init_model = self.model.state_dict()
        global_update = copy.deepcopy(init_model)
        excluded_num = int(len(models) * self.excluded_ratio)
        for key in init_model:
            temp = torch.stack([each_model[1][key] for each_model in models],
                               0)
            pos_largest, _ = torch.topk(temp, excluded_num, 0)
            neg_smallest, _ = torch.topk(-temp, excluded_num, 0)
            new_stacked = torch.cat([temp, -pos_largest,
                                     neg_smallest]).sum(0).float()
            new_stacked /= len(temp) - 2 * excluded_num
            global_update[key] = new_stacked
        return global_update
