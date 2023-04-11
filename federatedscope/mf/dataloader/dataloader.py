from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from numpy.random import shuffle

import numpy as np

import collections
import importlib

MFDATA_CLASS_DICT = {
    "vflmovielens1m": "VFLMovieLens1M",
    "vflmovielens10m": "VFLMovieLens10M",
    "hflmovielens1m": "HFLMovieLens1M",
    "hflmovielens10m": "HFLMovieLens10M"
}


def load_mf_dataset(config=None):
    """Return the dataset of matrix factorization

    Format:
        {
            'client_id': {
                'train': DataLoader(),
                'test': DataLoader(),
                'val': DataLoader()
            }
        }

    """
    if config.data.type.lower() in MFDATA_CLASS_DICT:
        # Dataset
        dataset = getattr(
            importlib.import_module("federatedscope.mf.dataset.movielens"),
            MFDATA_CLASS_DICT[config.data.type.lower()])(
                root=config.data.root,
                num_client=config.federate.client_num,
                train_portion=config.data.splits[0],
                download=True)
    else:
        raise NotImplementedError("Dataset {} is not implemented.".format(
            config.data.type))

    data_local_dict = collections.defaultdict(dict)
    for id_client, data in dataset.data.items():
        data_local_dict[id_client]["train"] = MFDataLoader(
            data["train"],
            shuffle=config.data.shuffle,
            batch_size=config.data.batch_size,
            drop_last=config.data.drop_last,
            theta=config.sgdmf.theta)
        data_local_dict[id_client]["test"] = MFDataLoader(
            data["test"],
            shuffle=False,
            batch_size=config.data.batch_size,
            drop_last=config.data.drop_last,
            theta=config.sgdmf.theta)

    # Modify config
    config.merge_from_list(['model.num_user', dataset.n_user])
    config.merge_from_list(['model.num_item', dataset.n_item])

    return data_local_dict, config


class MFDataLoader(object):
    """DataLoader for MF dataset

    Args:
        data (csc_matrix): sparse MF dataset
        batch_size (int): the size of batch data
        shuffle (bool): shuffle the dataset
        drop_last (bool): drop the last batch if True
        theta (int): the maximal number of ratings for each user
    """
    def __init__(self,
                 data: csc_matrix,
                 batch_size: int,
                 shuffle=True,
                 drop_last=False,
                 theta=None):
        super(MFDataLoader, self).__init__()
        self.dataset = self._trim_data(data, theta)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.n_row = self.dataset.shape[0]
        self.n_col = self.dataset.shape[1]
        self.n_rating = self.dataset.count_nonzero()

        self._idx_samples = None
        self._idx_cur = None

        self._reset()

    def _trim_data(self, data, theta=None):
        """Trim rating data by parameter theta (per-user privacy)

        Arguments:
            data (csc_matrix): the dataset
            theta (int): The maximal number of ratings for each user
        """
        if theta is None or theta <= 0:
            return data
        else:
            # Each user has at most $theta$ items
            dataset = data.tocoo()
            user2items = collections.defaultdict(list)
            for idx, user_id in enumerate(dataset.row):
                user2items[user_id].append(idx)
            # sample theta each
            idx_select = list()
            for items in user2items.values():
                if len(items) > theta:
                    idx_select += np.random.choice(items, theta,
                                                   replace=False).tolist()
                else:
                    idx_select += items
            dataset = coo_matrix(
                (dataset.data[idx_select],
                 (dataset.row[idx_select], dataset.col[idx_select])),
                shape=dataset.shape).tocsc()
            return dataset

    def _reset(self):
        self._idx_cur = 0
        if self._idx_samples is None:
            self._idx_samples = np.arange(self.n_rating)
        if self.shuffle:
            shuffle(self._idx_samples)

    def _sample_data(self, sampled_rating_idx):
        dataset = self.dataset.tocoo()
        data = dataset.data[sampled_rating_idx]
        rows = dataset.row[sampled_rating_idx]
        cols = dataset.col[sampled_rating_idx]
        return (rows, cols), data

    def __len__(self):
        """The number of batches within an epoch

        """
        if self.drop_last:
            return int(self.n_rating / self.batch_size)
        else:
            return int(self.n_rating / self.batch_size) + int(
                (self.n_rating % self.batch_size) != 0)

    def __next__(self, theta=None):
        """Get the next batch of data

        Args:
            theta (int): the maximal number of ratings for each user
        """
        idx_end = self._idx_cur + self.batch_size
        if self._idx_cur == len(
                self._idx_samples) or self.drop_last and idx_end > len(
                    self._idx_samples):
            raise StopIteration
        idx_end = min(idx_end, len(self._idx_samples))
        idx_choice_samples = self._idx_samples[self._idx_cur:idx_end]
        self._idx_cur = idx_end

        return self._sample_data(idx_choice_samples)

    def __iter__(self):
        self._reset()
        return self
