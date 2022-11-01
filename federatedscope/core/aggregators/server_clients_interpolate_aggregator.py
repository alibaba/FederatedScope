from federatedscope.core.aggregators import ClientsAvgAggregator


class ServerClientsInterpolateAggregator(ClientsAvgAggregator):
    """
    conduct aggregation by interpolating global model from server and \
    local models from clients
    """
    def __init__(self, model=None, device='cpu', config=None, beta=1.0):
        super(ServerClientsInterpolateAggregator,
              self).__init__(model, device, config)
        self.beta = beta  # the weight for local models used in interpolation

    def aggregate(self, agg_info):
        """
        Returns the aggregated value
        """
        models = agg_info["client_feedback"]
        global_model = self.model
        elem_each_client = next(iter(models))
        assert len(elem_each_client) == 2, f"Require (sample_size, " \
                                           f"model_para) tuple for each " \
                                           f"client, i.e., len=2, but got " \
                                           f"len={len(elem_each_client)}"
        avg_model_by_clients = self._para_weighted_avg(models)
        global_local_models = [((1 - self.beta), global_model.state_dict()),
                               (self.beta, avg_model_by_clients)]

        avg_model_by_interpolate = self._para_weighted_avg(global_local_models)
        return avg_model_by_interpolate
