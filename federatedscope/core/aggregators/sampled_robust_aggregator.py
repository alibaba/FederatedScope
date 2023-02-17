import copy
import torch
import random
from federatedscope.core.aggregators import ClientsAvgAggregator


class SampledrobustAggregator(ClientsAvgAggregator):
    """
    randomly sample a robust aggregator in each round of the FL course
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(SampledrobustAggregator, self).__init__(model, device, config)
        self.byzantine_node_num = config.aggregator.byzantine_node_num
        self.client_sampled_ratio = config.aggregator.client_sampled_ratio
        self.krum_agg_num = config.aggregator.sampled_robust_aggregator.krum_agg_num
        self.trimmedmean_alpha = config.aggregator.sampled_robust_aggregator.trimmedmean_excluded_ratio
        self.candidate=config.aggregator.sampled_robust_aggregator.candidate

        if 'krum' in self.candidate:
            assert 2 * self.byzantine_node_num + 2 < config.federate.client_num
        if 'trimmedmean' in self.candidate:
            assert 2 * self.trimmedmean_alpha < 1
        if 'bulyan' in self.candidate:
            assert 4 * self.byzantine_node_num + 3 <= config.federate.client_num

    def aggregate(self, agg_info):
        """
        To preform aggregation with Krum aggregation rule

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        models = agg_info["client_feedback"]

        rule_cur=random.choice(self.candidate)
        if rule_cur == 'krum':
            avg_model, this_round_id = self._para_avg_with_krum(models, agg_num=self.krum_agg_num)
        elif rule_cur == 'median':
            avg_model, this_round_id = self._aggre_with_median(models)
        elif rule_cur == 'trimmedmean':
            avg_model,this_round_id = self._aggre_with_trimmedmean(models)
        elif rule_cur == 'bulyan':
            avg_model,this_round_id = self._aggre_with_bulyan(models)

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
        closest_num = model_num - int(self.client_sampled_ratio*self.byzantine_node_num)

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
        this_round_id=list()
        models_para = [each_model[0][1] for each_model in models]
        krum_scores = self._calculate_score(models_para)
        index_order = torch.sort(krum_scores)[1].numpy()
        reliable_models = list()
        for number, index in enumerate(index_order):
            if number < agg_num:
                reliable_models.append(models[index][0])
                this_round_id.append(models[index][-1])
        return self._para_weighted_avg(models=reliable_models),this_round_id

    def _aggre_with_median(self, models):
        init_model = self.model.state_dict()
        global_update=copy.deepcopy(init_model)
        for key in init_model:
            temp=torch.stack([each_model[0][1][key] for each_model in models],0)
            temp_pos, _ =torch.median(temp,dim=0)
            temp_neg, _ =torch.median(-temp,dim=0)
            global_update[key]=(temp_pos-temp_neg)/2
        this_round_id=[each_model[-1] for each_model in models]
        return global_update, this_round_id

    def _aggre_with_trimmedmean(self, models):
        this_round_id=list()
        init_model = self.model.state_dict()
        global_update=copy.deepcopy(init_model)
        excluded_num=int(len(models)*self.trimmedmean_alpha)
        for key in init_model:
            temp=torch.stack([each_model[0][1][key] for each_model in models],0)
            pos_largest, _ = torch.topk(temp, excluded_num, 0)
            neg_smallest, _ = torch.topk(-temp, excluded_num, 0)
            new_stacked = torch.cat([temp, -pos_largest, neg_smallest]).sum(0).float()
            new_stacked /= len(temp) - 2 * excluded_num
            global_update[key] = new_stacked
        this_round_id=[each_model[-1] for each_model in models]
        return global_update,this_round_id

    # def _aggre_with_bulyan(self, models):
    #     '''
    #     Iteratively apply Krum to select \theta (\theta <= client_num- 
    #     2*self.byzantine_node_num) local models
    #     '''
    #     this_round_id=list()
    #     init_model = self.model.state_dict()
    #     global_update=copy.deepcopy(init_model)
    #     models_para = [each_model[0][1] for each_model in models]
    #     krum_scores = self._calculate_score(models_para)
    #     index_order = torch.sort(krum_scores)[1].numpy()
    #     reliable_models = list()
    #     for number, index in enumerate(index_order):
    #         if number < len(models) - int(2*self.client_sampled_ratio*self.byzantine_node_num):
    #             reliable_models.append(models[index][0])
    #             this_round_id.append(models[index][-1])
    #     '''
    #     sort parameter for each coordinate of the rest \theta reliable 
    #     local models, and find \gamma (gamma<\theta-2*self.byzantine_num) 
    #     parameters closest to the median to perform averaging
    #     '''
    #     gamma=len(reliable_models)-2*self.byzantine_node_num
    #     for key in init_model:
    #         tensor_shape=reliable_models[1][key].shape
    #         temp=torch.stack([each_model[1][key].view(-1) for each_model in reliable_models],0)
    #         d=list(temp.shape)[-1]
    #         param_median=torch.median(temp,dim=0).values
    #         diff= torch.abs(temp - param_median)
    #         cloest_indices = torch.argsort(diff, dim=0)[:gamma]
    #         closests = temp[cloest_indices, torch.arange(d)[None,:]]
    #         flatten_tensor=torch.mean(closests.float(), dim=0)
    #         global_update[key] = flatten_tensor.reshape(tensor_shape)
    #     return global_update, this_round_id