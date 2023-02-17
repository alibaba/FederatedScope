
import copy
import torch
from federatedscope.core.aggregators import ClientsAvgAggregator

class BulyanAggregator(ClientsAvgAggregator):
    """
    Implementation of Bulyan refers to `The Hidden Vulnerability 
    of Distributed Learning in Byzantium`
    [Mhamdi et al., 2018]
    (http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)

    It combines the MultiKrum aggregator and a variant of treamedmean aggregator
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(BulyanAggregator, self).__init__(model, device, config)
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        self.client_sampled_ratio = config.aggregator.client_sampled_ratio
        assert 4 * self.byzantine_node_num + 3 <= config.federate.client_num

    def aggregate(self, agg_info):
        """
        To preform aggregation with Median aggregation rule
        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]
        avg_model, this_round_id = self._aggre_with_bulyan(models)
        updated_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            updated_model[key] = init_model[key] + avg_model[key]
        return updated_model, this_round_id

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

    def _aggre_with_bulyan(self, models):
        '''
        apply MultiKrum to select \theta (\theta <= client_num- 
        2*self.byzantine_node_num) local models
        '''
        this_round_id=list()
        init_model = self.model.state_dict()
        global_update=copy.deepcopy(init_model)
        models_para = [each_model[0][1] for each_model in models]
        krum_scores = self._calculate_score(models_para)
        index_order = torch.sort(krum_scores)[1].numpy()
        reliable_models = list()
        for number, index in enumerate(index_order):
            if number < len(models) - int(2*self.client_sampled_ratio*self.byzantine_node_num):
                reliable_models.append(models[index][0])
                this_round_id.append(models[index][-1])
        
        '''
        sort parameter for each coordinate of the rest \theta reliable 
        local models, and find \gamma (gamma<\theta-2*self.byzantine_num) 
        parameters closest to the median to perform averaging
        '''
        gamma=len(reliable_models)-2*self.byzantine_node_num
        for key in init_model:
            tensor_shape=reliable_models[0][1][key].shape
            temp=torch.stack([each_model[1][key].view(-1) for each_model in reliable_models],0)
            d=list(temp.shape)[-1]
            param_median=torch.median(temp,dim=0).values
            diff= torch.abs(temp - param_median)
            cloest_indices = torch.argsort(diff, dim=0)[:gamma]
            closests = temp[cloest_indices, torch.arange(d)[None,:]]
            flatten_tensor=torch.mean(closests.float(), dim=0)
            global_update[key] = flatten_tensor.reshape(tensor_shape)
        return global_update, this_round_id