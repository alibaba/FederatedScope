from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):
    """
    The strategy of sampling
    """
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError


class UniformSampler(Sampler):
    """
    A stateless sampler that samples items uniformly
    """
    def __init__(self):
        super(UniformSampler, self).__init__()

    def sample(self, client_idle, size, *args,  **kwargs):
        """
        To sample clients
        """
        sampled_items = np.random.choice(client_idle,
                                         size=size,
                                         replace=False).tolist()
        return sampled_items


class GroupSampler(Sampler):
    """
    To grouply sample the clients based on their responsiveness (or other
    client information of the clients)
    """
    def __init__(self, client_info, bins=10):
        super(GroupSampler, self).__init__()
        self.bins = bins
        self.client_info = client_info
        self.candidate_iterator = self.partition()

    def partition(self):
        """
        To partition the clients into groups according to the client
        information

        Arguments:
        :returns: a iteration of candidates
        """
        # sort client_info by xx
        sorted_index = sorted(self.client_info.keys(), key=lambda x: self.client_info[x])
        # bin的长度
        num_per_bins = np.int(len(sorted_index) / self.bins)

        # grouped clients
        self.grouped_clients = np.split(
            sorted_index, np.cumsum([num_per_bins] * (self.bins - 1)))

        return self.permutation()

    def permutation(self):
        candidates = list()
        permutation = np.random.permutation(np.arange(self.bins))
        for i in permutation:
            np.random.shuffle(self.grouped_clients[i])
            candidates.extend(self.grouped_clients[i])

        return iter(candidates)

    def sample(self, clients_idle, size, perturb=False):
        """
        To sample clients
        """
        if self.candidate_iterator is None:
            self.partition()

        if perturb:
            self.candidate_iterator = self.permutation()

        sampled_clients = list()
        for i in range(size):
            # To find an idle client
            while True:
                try:
                    client_id = next(self.candidate_iterator)
                except StopIteration:
                    self.candidate_iterator = self.permutation()
                    client_id = next(self.candidate_iterator)

                if client_id in clients_idle:
                    break

            sampled_clients.append(client_id)

        return sampled_clients
