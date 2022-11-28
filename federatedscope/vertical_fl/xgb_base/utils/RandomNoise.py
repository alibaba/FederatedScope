import numpy as np


class random_noise:
    """
    Add random noises to the feature order to protect privacy.
    For more details, please see
        FederBoost: Private Federated Learning for GBDT
        (https://arxiv.org/pdf/2011.02796.pdf)
    """
    def __init__(self, epsilon=2, seed=123):
        self.epsilon = epsilon
        self.seed = seed

    def add_perm_noises_to_dict(self, old_dict, epsilon, bin_num):
        """
        Add dp noises to a dict whose items are lists(i.e., bins).
        For each item in a list,
            with prob. p = e^epsilon / (e^epsilon + bin_num - 1),
                it stays in the bin
            with prob. 1-p, it moves to another bin picked uniformly at random
        :param old_dict: dict whose values are lists
        :param epsilon: float
        :param bin_num: int
        :return: dict
        """
        new_dict = dict()
        for key in old_dict.keys():
            new_dict[key] = list()
        tmp = np.power(np.e, epsilon)
        p = tmp / (tmp + bin_num - 1)
        q = (1 - p) / (bin_num - 1)
        prob_list = [p] + [q] * (bin_num - 1)
        for key in old_dict.keys():
            for value in old_dict[key]:
                random_bin = np.random.choice(list(range(bin_num)),
                                              p=prob_list)
                if random_bin == 0:
                    new_dict[key].append(value)
                elif random_bin <= key:
                    new_dict[random_bin - 1].append(value)
                else:
                    new_dict[random_bin].append(value)
        # perturb the order of each list
        for key in new_dict.keys():
            new_dict[key] = np.random.permutation(new_dict[key])
        return new_dict

    def add_perm_noised_to_list_of_dict(self, old_list, epsilon, bin_num):
        """
        For each item in the list, do the above function
        :param old_list: list whose values are dict
        :param epsilon: float
        :param bin_num: int
        :return: list
        """
        length = len(old_list)
        new_list = list()
        for i in range(length):
            new_dict = self.add_perm_noises_to_dict(old_list[i], epsilon,
                                                    bin_num)
            new_list.append(new_dict)
        return new_list
