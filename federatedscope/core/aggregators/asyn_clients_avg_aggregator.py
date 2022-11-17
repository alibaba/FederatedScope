import copy
import torch
from federatedscope.core.aggregators import ClientsAvgAggregator


class AsynClientsAvgAggregator(ClientsAvgAggregator):
    """
    The aggregator used in asynchronous training, which discounts the \
    staled model updates
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(AsynClientsAvgAggregator, self).__init__(model, device, config)

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        staleness = [x[1]
                     for x in agg_info['staleness']]  # (client_id, staleness)
        avg_model = self._para_weighted_avg(models,
                                            recover_fun=recover_fun,
                                            staleness=staleness)

        # When using asynchronous training, the return feedback is model delta
        # rather than the model param
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model

    def discount_func(self, staleness):
        """
        Served as an example, we discount the model update with staleness tau \
        as: ``(1.0/((1.0+\tau)**factor))``, \
        which has been used in previous studies such as FedAsync ( \
        Asynchronous Federated Optimization) and FedBuff \
        (Federated Learning with Buffered Asynchronous Aggregation).
        """
        return (1.0 /
                ((1.0 + staleness)**self.cfg.asyn.staleness_discount_factor))

    def _para_weighted_avg(self, models, recover_fun=None, staleness=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        for key in avg_model:
            for i in range(len(models)):
                local_sample_size, local_model = models[i]

                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                else:
                    weight = local_sample_size / training_set_size

                assert staleness is not None
                weight *= self.discount_func(staleness[i])
                if isinstance(local_model[key], torch.Tensor):
                    local_model[key] = local_model[key].float()
                else:
                    local_model[key] = torch.FloatTensor(local_model[key])

                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

        return avg_model
