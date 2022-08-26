from abc import ABC, abstractmethod
import numpy as np
try:
    import torch
except ImportError:
    torch = None


class SecretSharing(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def secret_split(self, secret):
        pass

    @abstractmethod
    def secret_reconstruct(self, secret_seq):
        pass


class AdditiveSecretSharing(SecretSharing):
    """
    AdditiveSecretSharing class, which can split a number into frames and
    recover it by summing up
    """
    def __init__(self, shared_party_num, size=60):
        super(SecretSharing, self).__init__()
        assert shared_party_num > 1, "AdditiveSecretSharing require " \
                                     "shared_party_num > 1"
        self.shared_party_num = shared_party_num
        self.maximum = 2**size
        self.mod_number = 2 * self.maximum + 1
        self.epsilon = 1e8
        self.mod_funs = np.vectorize(lambda x: x % self.mod_number)
        self.float2fixedpoint = np.vectorize(self._float2fixedpoint)
        self.fixedpoint2float = np.vectorize(self._fixedpoint2float)
        self.field_size = 10000

    def secret_split(self, secret):
        """
        To split the secret into frames according to the shared_party_num
        """
        if isinstance(secret, dict):
            secret_list = [dict() for _ in range(self.shared_party_num)]
            for key in secret:
                for idx, each in enumerate(self.secret_split(secret[key])):
                    secret_list[idx][key] = each
            return secret_list

        if isinstance(secret, list) or isinstance(secret, np.ndarray):
            secret = np.asarray(secret)
            shape = [self.shared_party_num - 1] + list(secret.shape)
        elif isinstance(secret, torch.Tensor):
            secret = secret.numpy()
            shape = [self.shared_party_num - 1] + list(secret.shape)
        else:
            shape = [self.shared_party_num - 1]
        secret_seq = np.random.uniform(low=0, high=100, size=shape)
        # last_seq = self.mod_funs(secret - self.mod_funs(np.sum(secret_seq,
        # axis=0)))
        last_seq = secret - np.sum(secret_seq, axis=0)

        secret_seq = np.append(secret_seq,
                               np.expand_dims(last_seq, axis=0),
                               axis=0)
        # secret_seq = [self.fixedpoint2float(x) for x in secret_seq]
        return secret_seq

    def secret_reconstruct(self, secret_seq):
        """
        To recover the secret
        """
        assert len(secret_seq) == self.shared_party_num
        merge_model = secret_seq[0].copy()
        if isinstance(merge_model, dict):
            for key in merge_model:
                for idx in range(len(secret_seq)):
                    if idx == 0:
                        merge_model[key] = secret_seq[idx][key].copy()
                    else:
                        merge_model[key] += secret_seq[idx][key]
        else:
            secret_seq = [np.asarray(x) for x in secret_seq]
            for idx in range(len(secret_seq)):
                if idx == 0:
                    merge_model = secret_seq[idx].copy()
                else:
                    merge_model += secret_seq[idx]
            # merge_model = self.fixedpoint2float(merge_model)
        return merge_model

    def _float2fixedpoint(self, x):
        # x = self.fixedpoint2float(x)
        # x = round(x * self.epsilon, 0)
        # assert abs(x) < self.maximum
        return x * 100

    def _fixedpoint2float(self, x):
        x = x % self.mod_number
        if x > self.maximum:
            return -1 * (self.mod_number - x) / self.epsilon
        else:
            return x / self.epsilon
