import numpy as np


class Opboost_noise:
    """
        Add random noises to the feature order to protect privacy.
        For more details, please see
            OpBoost- A Vertical Federated Tree Boosting Framework Based on
                Order-Preserving Desensitization.pdf
            (https://arxiv.org/pdf/2210.01318.pdf)
        """
    def __init__(self, epsilon=1, epsilon_prt=1, lower=1, upper=10):
        self.epsilon = epsilon
        self.epsilon_prt = epsilon_prt
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

    def global_map(self, data):
        data = self.data_preprocess(data)
        new_data = data.copy()
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                prob_list = []
                denominator = np.sum(
                    np.exp(
                        -np.abs(data[i][j] -
                                np.array(range(self.lower, self.upper + 1))) *
                        self.epsilon / 2))
                for k in range(self.lower, self.upper + 1):
                    prob_list.append(
                        np.exp(-np.abs(data[i][j] - k) * self.epsilon / 2) /
                        denominator)
                new_data[i][j] = np.random.choice(list(
                    range(self.lower, self.upper + 1)),
                                                  p=prob_list)
        feature_order = self.order_feature(new_data)
        return feature_order
