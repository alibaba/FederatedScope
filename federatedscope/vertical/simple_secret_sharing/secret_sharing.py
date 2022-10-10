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
    def __init__(self, shared_party_num, size=30):
        super(SecretSharing, self).__init__()
        assert shared_party_num > 1, "AdditiveSecretSharing require " \
                                     "shared_party_num > 1"
        self.shared_party_num = shared_party_num
        self.maximum = 2**size
        self.mod_number = 2**(2 * size)
        self.epsilon = 1e8
        self.mod_funs = np.vectorize(lambda x: x % self.mod_number)
        self.float2fixedpoint = np.vectorize(self._float2fixedpoint)
        self.fixedpoint2float = np.vectorize(self._fixedpoint2float)
        self.upgrade = np.vectorize(self._upgrade)
        self.downgrade = np.vectorize(self._downgrade)

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
        last_seq = self.mod_funs(secret -
                                 self.mod_funs(np.sum(secret_seq, axis=0)))

        secret_seq = np.append(secret_seq,
                               np.expand_dims(last_seq, axis=0),
                               axis=0)
        return secret_seq

    def secret_split_for_piece_of_ss(self, secret):
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
        secret_seq = np.random.randint(low=0, high=self.mod_number, size=shape)
        last_seq = self.mod_funs(secret -
                                 self.mod_funs(np.sum(secret_seq, axis=0)))

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
                        merge_model[key] = secret_seq[idx][key].copy()
                    else:
                        merge_model[key] += secret_seq[idx][key]
                merge_model[key] = self.fixedpoint2float(merge_model[key])
        else:
            secret_seq = [np.asarray(x) for x in secret_seq]
            for idx in range(len(secret_seq)):
                if idx == 0:
                    merge_model = secret_seq[idx].copy()
                else:
                    merge_model += secret_seq[idx]
            merge_model = self.fixedpoint2float(merge_model)
        return merge_model

    def secret_reconstruct_for_ss_pieces(self, secret_seq):
        """
        To recover a piece of ss pieces
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
                merge_model[key] = self.fixedpoint2float(merge_model[key])
        else:
            secret_seq = [np.asarray(x) for x in secret_seq]
            for idx in range(len(secret_seq)):
                if idx == 0:
                    merge_model = secret_seq[idx].copy()
                else:
                    merge_model += secret_seq[idx]
            merge_model = self.mod_funs(merge_model)
        return merge_model

    def const_add_fixedpoint(self, c, x):
        up_c = self._upgrade(c)
        res = (up_c + x) % self.mod_number
        return res

    def const_mul_fixedpoint(self, c, x):
        up_c = self.upgrade(c)
        res = self.downgrade(up_c * x)
        return res

    def mod_add(self, *args):
        # sum the values in an array
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            res = 0
            for x in args[0]:
                res += x.item()
                res = res % self.mod_number
            return res
        # sum the values in a list
        if len(args) == 1 and isinstance(args[0], list):
            res = 0
            for x in args[0]:
                res += x
                res = res % self.mod_number
            return res
        # sum an integer with a list or an array
        if len(args) == 2:
            if (isinstance(args[0], int) or isinstance(
                    args[0], np.int64)) and not isinstance(args[1], int):
                res = [(int(args[0]) + x) % self.mod_number for x in args[1]]
                return res
            elif (isinstance(args[1], int) or isinstance(
                    args[1], np.int64)) and not isinstance(args[0], int):
                res = [(args[1] + x) % self.mod_number for x in args[0]]
                return res
        # sum several lists
        if isinstance(args[0], np.ndarray) or isinstance(args[0], list):
            n = len(args[0])
            num = len(args)
            res = [0] * n
            for i in range(n):
                for j in range(num):
                    res[i] += args[j][i].item()
                    res[i] = res[i] % self.mod_number
            return np.asarray(res)
        # sum all the integers in args
        if isinstance(args[0], int) or isinstance(args[0], np.int64):
            res = 0
            for x in args:
                res += x
                res = res % self.mod_number
            return res
        # sum dict_values
        else:
            l_tmp = list(args[0])
            num = len(l_tmp)
            # a list of integers
            if isinstance(l_tmp[0], int):
                res = 0
                for x in l_tmp:
                    res += x
                    res = res % self.mod_number
                return res
            # a list of lists
            else:
                n = len(l_tmp[0])
                res = [0] * n
                for i in range(n):
                    for j in range(num):
                        res[i] += l_tmp[j][i]
                        res[i] = res[i] % self.mod_number
                return np.asarray(res)

    def _float2fixedpoint(self, x):
        x = round(x * self.epsilon)
        assert abs(x) < self.maximum
        return x % self.mod_number

    def _fixedpoint2float(self, x):
        x = x % self.mod_number
        if x > self.mod_number / 2:  # self.maximum:
            return -1 * (self.mod_number - x) / self.epsilon
        else:
            return x / self.epsilon

    def _upgrade(self, x):
        x = round(x * self.epsilon)
        assert abs(x) < self.maximum
        return x

    def _downgrade(self, x):
        x = round(x / self.epsilon)
        x = x % self.mod_number
        return x
