import os
import pickle
import logging

from torchvision.datasets.utils import check_integrity, \
    download_and_extract_archive, calculate_md5
import pandas as pd
from numpy.random import shuffle
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
import numpy as np

logger = logging.getLogger(__name__)


class VMFDataset:
    """Dataset of matrix factorization task in vertical federated learning.

    """
    def _split_n_clients_rating(self, ratings: csc_matrix, num_client: int,
                                test_portion: float):
        id_item = np.arange(self.n_item)
        shuffle(id_item)
        items_per_client = np.array_split(id_item, num_client)
        data = dict()
        for clientId, items in enumerate(items_per_client):
            client_ratings = ratings[:, items]
            train_ratings, test_ratings = self._split_train_test_ratings(
                client_ratings, test_portion)
            data[clientId + 1] = {"train": train_ratings, "test": test_ratings}
        self.data = data


class HMFDataset:
    """Dataset of matrix factorization task in horizontal federated learning.

    """
    def _split_n_clients_rating(self, ratings: csc_matrix, num_client: int,
                                test_portion: float):
        id_user = np.arange(self.n_user)
        shuffle(id_user)
        users_per_client = np.array_split(id_user, num_client)
        data = dict()
        for cliendId, users in enumerate(users_per_client):
            client_ratings = ratings[users, :]
            train_ratings, test_ratings = self._split_train_test_ratings(
                client_ratings, test_portion)
            data[cliendId + 1] = {"train": train_ratings, "test": test_ratings}
        self.data = data


class MovieLensData(object):
    """Download and split MF datasets

    Arguments:
        root (string): the path of data
        num_client (int): the number of clients
        train_portion (float): the portion of training data
        download (bool): indicator to download dataset
    """
    def __init__(self, root, num_client, train_portion=0.9, download=True):
        super(MovieLensData, self).__init__()

        self.root = root
        self.data = None

        self.n_user = None
        self.n_item = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." +
                               "You can use download=True to download it")

        ratings = self._load_meta()
        self._split_n_clients_rating(ratings, num_client, 1 - train_portion)

    def _split_n_clients_rating(self, ratings: csc_matrix, num_client: int,
                                test_portion: float):
        id_item = np.arange(self.n_item)
        shuffle(id_item)
        items_per_client = np.array_split(id_item, num_client)
        data = dict()
        for clientId, items in enumerate(items_per_client):
            client_ratings = ratings[:, items]
            train_ratings, test_ratings = self._split_train_test_ratings(
                client_ratings, test_portion)
            data[clientId + 1] = {"train": train_ratings, "test": test_ratings}
        self.data = data

    def _split_train_test_ratings(self, ratings: csc_matrix,
                                  test_portion: float):
        n_ratings = ratings.count_nonzero()
        id_test = np.random.choice(n_ratings,
                                   int(n_ratings * test_portion),
                                   replace=False)
        id_train = list(set(np.arange(n_ratings)) - set(id_test))

        ratings = ratings.tocoo()
        test = coo_matrix((ratings.data[id_test],
                           (ratings.row[id_test], ratings.col[id_test])),
                          shape=ratings.shape)
        train = coo_matrix((ratings.data[id_train],
                            (ratings.row[id_train], ratings.col[id_train])),
                           shape=ratings.shape)

        train_ratings, test_ratings = train.tocsc(), test.tocsc()
        return train_ratings, test_ratings

    def _load_meta(self):
        meta_path = os.path.join(self.root, self.base_folder, "ratings.pkl")
        if not os.path.exists(meta_path):
            logger.info("Processing data into {} parties.")
            fpath = os.path.join(self.root, self.base_folder, self.filename,
                                 self.raw_file)
            data = pd.read_csv(fpath,
                               sep="::",
                               engine="python",
                               usecols=[0, 1, 2],
                               names=["userId", "movieId", "rating"],
                               dtype={
                                   "userId": np.int32,
                                   "movieId": np.int32,
                                   "rating": np.float32
                               })
            # Map idx
            unique_id_item, unique_id_user = np.sort(
                data["movieId"].unique()), np.sort(data["userId"].unique())
            n_item, n_user = len(unique_id_item), len(unique_id_user)
            mapping_item, mapping_user = {
                mid: idx
                for idx, mid in enumerate(unique_id_item)
            }, {mid: idx
                for idx, mid in enumerate(unique_id_user)}

            row = [mapping_user[mid] for _, mid in data["userId"].iteritems()]
            col = [mapping_item[mid] for _, mid in data["movieId"].iteritems()]

            ratings = coo_matrix((data["rating"], (row, col)),
                                 shape=(n_user, n_item))
            ratings = ratings.tocsc()

            with open(meta_path, 'wb') as f:
                pickle.dump(ratings, f)
            logger.info("Done.")
        else:
            with open(meta_path, 'rb') as f:
                ratings = pickle.load(f)

        self.n_user, self.n_item = ratings.shape
        return ratings

    def _check_integrity(self):
        fpath = os.path.join(self.root, self.base_folder, self.filename,
                             self.raw_file)
        return check_integrity(fpath, self.raw_file_md5)

    def download(self):
        if self._check_integrity():
            logger.info("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url,
                                     os.path.join(self.root, self.base_folder),
                                     filename=self.url.split('/')[-1],
                                     md5=self.zip_md5)


class MovieLens1M(MovieLensData):
    """MoviesLens 1M Dataset
    (https://grouplens.org/datasets/movielens)

    Format:
        UserID::MovieID::Rating::Timestamp

    Arguments:
        root (str): Root directory of dataset where directory
            ``MoviesLen1M`` exists or will be saved to if download is
            set to True.
        config (callable): Parameters related to matrix factorization.
        train_size (float, optional): The proportion of training data.
        test_size (float, optional): The proportion of test data.
        download  (bool, optional): If true, downloads the dataset from the
        internet and puts it in root directory. If dataset is already
        downloaded, it is not downloaded again.

    """
    base_folder = 'MovieLens1M'
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    filename = "ml-1m"
    zip_md5 = "c4d9eecfca2ab87c1945afe126590906"
    raw_file = "ratings.dat"
    raw_file_md5 = "a89aa3591bc97d6d4e0c89459ff39362"


class MovieLens10M(MovieLensData):
    """MoviesLens 10M Dataset
    (https://grouplens.org/datasets/movielens)

    Format:
        UserID::MovieID::Rating::Timestamp

    Arguments:
        root (str): Root directory of dataset where directory
            ``MoviesLen1M`` exists or will be saved to if download is
            set to True.
        config (callable): Parameters related to matrix factorization.
        train_size (float, optional): The proportion of training data.
        test_size (float, optional): The proportion of test data.
        download  (bool, optional): If true, downloads the dataset from the
        internet and
            puts it in root directory. If dataset is already downloaded,
            it is not
            downloaded again.

    """
    base_folder = 'MovieLens10M'
    url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
    filename = "ml-10M100K"

    zip_md5 = "ce571fd55effeba0271552578f2648bd"
    raw_file = "ratings.dat"
    raw_file_md5 = "3f317698625386f66177629fa5c6b2dc"


class VFLMovieLens1M(MovieLens1M, VMFDataset):
    """MovieLens1M dataset in VFL setting

    """
    pass


class HFLMovieLens1M(MovieLens1M, HMFDataset):
    """MovieLens1M dataset in HFL setting

    """
    pass


class VFLMovieLens10M(MovieLens10M, VMFDataset):
    """MovieLens10M dataset in VFL setting

    """
    pass


class HFLMovieLens10M(MovieLens10M, HMFDataset):
    """MovieLens10M dataset in HFL setting

    """
    pass
