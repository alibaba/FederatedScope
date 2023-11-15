import copy
import torch
from federatedscope.core.aggregators import ClientsAvgAggregator


class KrumAggregator(ClientsAvgAggregator):
    """
    Implementation of Krum/multi-Krum refer to `Machine learning with
    adversaries: Byzantine tolerant gradient descent`
    [Blanchard P et al., 2017]
    (https://proceedings.neurips.cc/paper/2017/hash/
    f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(KrumAggregator, self).__init__(model, device, config)
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        self.krum_agg_num = config.aggregator.BFT_args.krum_agg_num
        assert 2 * self.byzantine_node_num + 2 < config.federate.client_num, \
            "it should be satisfied that 2*byzantine_node_num + 2 < client_num"

    def aggregate(self, agg_info):
        """
        To preform aggregation with Krum aggregation rule

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        avg_model = self._para_avg_with_krum(models, agg_num=self.krum_agg_num)

        # When using Krum/multi-Krum aggregation, the return feedback is model
        # delta rather than the model param
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model

    def _calculate_distance(self, model_a, model_b):
        """
        Calculate the Euclidean distance between two given model para delta
        """
        distance = 0.0

        for key in model_a:
            if isinstance(model_a[key], torch.Tensor):
                model_a[key] = model_a[key].float()
                model_b[key] = model_b[key].float()
            else:
                model_a[key] = torch.FloatTensor(model_a[key])
                model_b[key] = torch.FloatTensor(model_b[key])

            distance += torch.dist(model_a[key], model_b[key], p=2)
        return distance

    def _calculate_score(self, models):
        """
        Calculate Krum scores
        """
        model_num = len(models)
        closest_num = model_num - self.byzantine_node_num - 2

        distance_matrix = torch.zeros(model_num, model_num)
        for index_a in range(model_num):
            for index_b in range(index_a, model_num):
                if index_a == index_b:
                    distance_matrix[index_a, index_b] = float('inf')
                else:
                    distance_matrix[index_a, index_b] = distance_matrix[
                        index_b, index_a] = self._calculate_distance(
                            models[index_a], models[index_b])

        sorted_distance = torch.sort(distance_matrix)[0]
        krum_scores = torch.sum(sorted_distance[:, :closest_num], axis=-1)
        return krum_scores

    def _para_avg_with_krum(self, models, agg_num=1):

        # each_model: (sample_size, model_para)
        models_para = [each_model[1] for each_model in models]
        krum_scores = self._calculate_score(models_para)
        index_order = torch.sort(krum_scores)[1].numpy()
        reliable_models = list()
        for number, index in enumerate(index_order):
            if number < agg_num:
                reliable_models.append(models[index])

        return self._para_weighted_avg(models=reliable_models)
