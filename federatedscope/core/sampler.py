from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):
    """
    The strategies of sampling clients for each training round

    Arguments:
        client_state: a dict to manager the state of clients (idle or busy)
    """
    def __init__(self, client_num):
        self.client_state = np.asarray([1] * (client_num + 1))
        # Set the state of server (index=0) to 'working'
        self.client_state[0] = 0

    @abstractmethod
    def sample(self, size):
        raise NotImplementedError

    def change_state(self, indices, state):
        """
        To modify the state of clients (idle or working)
        """
        if isinstance(indices, list) or isinstance(indices, np.ndarray):
            all_idx = indices
        else:
            all_idx = [indices]
        for idx in all_idx:
            if state in ['idle', 'seen']:
                self.client_state[idx] = 1
            elif state in ['working', 'unseen']:
                self.client_state[idx] = 0
            else:
                raise ValueError(
                    f"The state of client should be one of "
                    f"['idle', 'working', 'unseen], but got {state}")


class UniformSampler(Sampler):
    """
    To uniformly sample the clients from all the idle clients
    """
    def __init__(self, client_num):
        super(UniformSampler, self).__init__(client_num)

    def sample(self, size):
        """
        To sample clients
        """
        idle_clients = np.nonzero(self.client_state)[0]
        sampled_clients = np.random.choice(idle_clients,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working')
        return sampled_clients


class GroupSampler(Sampler):
    """
    To grouply sample the clients based on their responsiveness (or other
    client information of the clients)
    """
    def __init__(self, client_num, client_info, bins=10):
        super(GroupSampler, self).__init__(client_num)
        self.bins = bins
        self.update_client_info(client_info)
        self.candidate_iterator = self.partition()

    def update_client_info(self, client_info):
        """
        To update the client information
        """
        self.client_info = np.asarray(
            [1.0] + [x for x in client_info
                     ])  # client_info[0] is preversed for the server
        assert len(self.client_info) == len(
            self.client_state
        ), "The first dimension of client_info is mismatched with client_num"

    def partition(self):
        """
        To partition the clients into groups according to the client
        information

        Arguments:
        :returns: a iteration of candidates
        """
        sorted_index = np.argsort(self.client_info)
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

    def sample(self, size, shuffle=False):
        """
        To sample clients
        """
        if shuffle:
            self.candidate_iterator = self.permutation()

        sampled_clients = list()
        for i in range(size):
            # To find an idle client
            while True:
                try:
                    item = next(self.candidate_iterator)
                except StopIteration:
                    self.candidate_iterator = self.permutation()
                    item = next(self.candidate_iterator)

                if self.client_state[item] == 1:
                    break

            sampled_clients.append(item)
            self.change_state(item, 'working')

        return sampled_clients


class ResponsivenessRealtedSampler(Sampler):
    """
    To sample the clients based on their responsiveness (or other information
    of clients)
    """
    def __init__(self, client_num, client_info):
        super(ResponsivenessRealtedSampler, self).__init__(client_num)
        self.update_client_info(client_info)

    def update_client_info(self, client_info):
        """
        To update the client information
        """
        self.client_info = np.asarray(
            [1.0] + [np.sqrt(x) for x in client_info
                     ])  # client_info[0] is preversed for the server
        assert len(self.client_info) == len(
            self.client_state
        ), "The first dimension of client_info is mismatched with client_num"

    def sample(self, size):
        """
        To sample clients
        """
        idle_clients = np.nonzero(self.client_state)[0]
        client_info = self.client_info[idle_clients]
        client_info = client_info / np.sum(client_info, keepdims=True)
        sampled_clients = np.random.choice(idle_clients,
                                           p=client_info,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working')
        return sampled_clients
