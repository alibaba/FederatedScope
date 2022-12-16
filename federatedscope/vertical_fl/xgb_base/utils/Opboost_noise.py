import numpy as np


class Opboost_noise:
    """
        Add random noises to the feature order to protect privacy.
        For more details, please see
            OpBoost- A Vertical Federated Tree Boosting Framework Based on
                Order-Preserving Desensitization.pdf
            (https://arxiv.org/pdf/2210.01318.pdf)
        """
    def __init__(self, lower=1, upper=100):
        self.feature_order = None
        self.lower = lower
        self.upper = upper

    def order_feature(self, data):
        feature_order = [0] * data.shape[1]
        for j in range(data.shape[1]):
            feature_order[j] = data[:, j].argsort()
        return feature_order

    def data_preprocess(self, data):
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        # avoid the case that data_max[i] == data_min[i],
        #   which the denominator will eq 0
        for i in range(data.shape[1]):
            if data_max[i] == data_min[i]:
                data_max[i] += 1
        processed_data = np.round(self.lower + (data - data_min) /
                                  (data_max - data_min) *
                                  (self.upper - self.lower))
        return processed_data

    def random_sample(self, x, epsilon, lower, upper):
        prob_list = []
        denominator = np.sum(
            np.exp(-np.abs(x - np.array(range(lower, upper + 1))) * epsilon /
                   2))
        for k in range(lower, upper + 1):
            prob_list.append(
                np.exp(-np.abs(x - k) * epsilon / 2) / denominator)
        res = np.random.choice(list(range(lower, upper + 1)), p=prob_list)
        return res

    def global_map(self, data, epsilon):
        data = self.data_preprocess(data)
        new_data = data.copy()
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                new_data[i][j] = self.random_sample(data[i][j], epsilon,
                                                    self.lower, self.upper)
        return new_data

    def global_order(self, data, epsilon=1):
        new_data = self.global_map(data, epsilon)
        feature_order = self.order_feature(new_data)
        return feature_order

    def adj_map(self, data, epsilon_prt, epsilon_ner, partition_num):
        quantiles = np.linspace(0, 100, partition_num + 1)
        partition_edges = np.round(
            np.asarray(
                np.percentile(list(range(self.lower - 1, self.upper + 1)),
                              quantiles)))
        partition_edges = [int(x) for x in partition_edges]
        data = self.global_map(data, epsilon_prt)
        new_data = data.copy()
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                for k in range(partition_num):
                    if partition_edges[k] < data[i][j] <= partition_edges[k +
                                                                          1]:
                        new_part = self.random_sample(k, epsilon_prt, 0,
                                                      partition_num - 1)
                        new_data[i][j] = self.random_sample(
                            data[i][j], epsilon_ner,
                            partition_edges[new_part] + 1,
                            partition_edges[new_part + 1])
        return new_data

    def adj_order(self, data, epsilon_prt=1, epsilon_ner=1, partition_num=10):
        new_data = self.adj_map(data, epsilon_prt, epsilon_ner, partition_num)
        feature_order = self.order_feature(new_data)
        return feature_order
