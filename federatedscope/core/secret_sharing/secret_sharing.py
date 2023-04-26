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

        secret = self.float2fixedpoint(secret)
        secret_seq = np.random.randint(low=0, high=self.mod_number, size=shape)
        # last_seq =
        #   self.mod_funs(secret - self.mod_funs(np.sum(secret_seq, axis=0)))
        last_seq = self.mod_funs(
            secret - self.mod_funs(np.sum(secret_seq, axis=0))).astype(int)

        secret_seq = np.append(secret_seq,
                               np.expand_dims(last_seq, axis=0),
                               axis=0)
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
                        merge_model[key] = secret_seq[idx][key]
                    else:
                        merge_model[key] += secret_seq[idx][key]
                merge_model[key] = self.fixedpoint2float(merge_model[key])
        # if merge_model is an ndarray or a list
        else:
            for idx in range(len(secret_seq)):
                if idx == 0:
                    merge_model = secret_seq[idx].copy()
                else:
                    merge_model += secret_seq[idx]
            merge_model = self.fixedpoint2float(merge_model)

        return merge_model

    def _float2fixedpoint(self, x):
        x = round(x * self.epsilon, 0)
        assert abs(x) < self.maximum
        return x % self.mod_number

    def _fixedpoint2float(self, x):
        x = x % self.mod_number
        if x > self.maximum:
            return -1 * (self.mod_number - x) / self.epsilon
        else:
            return x / self.epsilon


class MultiplicativeSecretSharing(AdditiveSecretSharing):
    """
    AdditiveSecretSharing class, which can split a number into frames and
    recover it by summing up
    """
    def __init__(self, shared_party_num, size=60):
        super().__init__(shared_party_num, size)
        self.maximum = 2**size
        self.mod_number = 2 * self.maximum + 1
        self.epsilon = 1e8

    def secret_split(self, secret, cls=None):
        """
        To split the secret into frames according to the shared_party_num
        """
        if isinstance(secret, dict):
            secret_list = [dict() for _ in range(self.shared_party_num)]
            for key in secret:
                for idx, each in enumerate(
                        self.secret_split(secret[key], cls=cls)):
                    secret_list[idx][key] = each
            return secret_list

        if isinstance(secret, list) or isinstance(secret, np.ndarray):
            secret = np.asarray(secret).astype(int)
            shape = [self.shared_party_num - 1] + list(secret.shape)
        elif isinstance(secret, torch.Tensor):
            secret = secret.numpy()
            shape = [self.shared_party_num - 1] + list(secret.shape)
        else:
            shape = [self.shared_party_num - 1]

        if cls is None:
            secret = self.float2fixedpoint(secret)
        secret_seq = np.random.randint(low=0, high=self.mod_number, size=shape)
        # last_seq =
        #   self.mod_funs(secret - self.mod_funs(np.sum(secret_seq, axis=0)))
        last_seq = self.mod_funs(
            secret - self.mod_funs(np.sum(secret_seq, axis=0))).astype(int)

        secret_seq = np.append(secret_seq,
                               np.expand_dims(last_seq, axis=0),
                               axis=0)
        return secret_seq

    def secret_add_lists(self, args):
        # args is a list
        #   whose last element is a list consisting of secret pieces
        # TODO: add the condition that all elements in args are numbers
        for i in range(len(args) - 1):
            # if isinstance(args[i], int) or isinstance(args[i], np.int64):
            if not isinstance(args[i], list) and not isinstance(
                    args[i], np.ndarray):
                args[i] = [args[i]] * len(args[-1])
        return self.mod_funs(np.sum(args, axis=0))
        # TODO: in the future, when involve large numbers, numpy may overflow,
        #  thus, the following would work
        # n = len(args[0])
        # num = len(args)
        # res = [0] * n
        # for i in range(n):
        #     for j in range(num):
        #         res[i] += args[j][i]
        #         res[i] = res[i] % self.mod_number
        # return np.asarray(res)

    def secret_ndarray_star_ndarray(self, arr1, arr2):
        # return a list whose i-th elements equals to
        # the product of the i-th elements of arr1 and arr2
        # where arr1 and arr2 are both secret pieces
        if isinstance(arr1, int) or isinstance(arr1, np.int64):
            arr1 = [arr1] * len(arr2)
        if isinstance(arr2, int) or isinstance(arr2, np.int64):
            arr2 = [arr2] * len(arr1)
        n = len(arr1)
        res = [0] * n
        for i in range(n):
            res[i] = (arr1[i].item() * arr2[i].item()) % self.mod_number
        return np.asarray(res)

    def beaver_triple(self, *args):
        a = np.random.randint(0, self.mod_number, args).astype(int)
        b = np.random.randint(0, self.mod_number, args).astype(int)

        a_list = []
        b_list = []
        c = [(a[i].item() * b[i].item()) % self.mod_number
             for i in range(len(a))]
        c_list = []
        for i in range(self.shared_party_num - 1):
            a_tmp = np.random.randint(0, self.mod_number, args)
            a_list.append(a_tmp)
            a -= a_tmp
            a = a % self.mod_number
            b_tmp = np.random.randint(0, self.mod_number, args)
            b_list.append(b_tmp)
            b -= b_tmp
            b = b % self.mod_number
            c_tmp = np.random.randint(0, self.mod_number, args)
            c_list.append(c_tmp)
            c -= c_tmp
            c = c % self.mod_number
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
        return a_list, b_list, c_list
