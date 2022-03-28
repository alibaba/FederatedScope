import collections
import os
import pickle

import torch.nn
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import shuffle
import numpy as np
import math


class MFDataset(Dataset):
    def __init__(self, data):
        super(MFDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MovieLens1M(object):
    """MoviesLens 1M Dataset
    (https://grouplens.org/datasets/movielens)

    Format:
        UserID::MovieID::Rating::Timestamp

    Arguments:
        root (str):
        num_client (int):
        rebuild (bool): if rebuild t

    """
    name = 'MoviesLens1M'
    raw_file = "ratings.dat"
    resources = [("https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                  "c4d9eecfca2ab87c1945afe126590906")]

    def __init__(self,
                 config,
                 root,
                 num_client,
                 train_size=0.9,
                 test_size=0.1,
                 theta=5,
                 rebuild=False):
        super(MovieLens1M, self).__init__()

        self.root = root
        self.theta = theta
        self.num_client = num_client
        self.processed_folder = os.path.join(self.root, self.name)

        self.data = None

        assert train_size + test_size == 1
        self._parse_data(config, train_size, rebuild)

    def _parse_data(self, config, train_size, rebuild) -> None:
        path_raw_file = os.path.join(self.processed_folder, 'ml-1m',
                                     self.raw_file)
        if not os.path.exists(path_raw_file):
            self._download()

        path_data = os.path.join(
            self.processed_folder,
            "data_{}_client_{}_split.pl".format(self.num_client, train_size))
        if os.path.exists(path_data) and not rebuild:
            file = pickle.load(open(path_data, 'rb'))
            self.data = file["data"]
            self.num_user = file["num_user"]
            self.num_movie = file["num_movie"]
        else:
            raw_data = pd.read_csv(path_raw_file,
                                   sep="::",
                                   engine="python",
                                   usecols=[0, 1, 2],
                                   names=["userid", "movieid", "rating"],
                                   dtype={
                                       "userid": np.int32,
                                       "movieid": np.int32,
                                       "rating": np.float32
                                   })

            user_unique = np.sort(raw_data["userid"].unique())
            movie_unique = np.sort(raw_data["movieid"].unique())

            self.num_user, self.num_movie = len(user_unique), len(movie_unique)
            map_userid = dict(zip(user_unique, np.arange(self.num_user)))
            map_movieid = dict(zip(movie_unique, np.arange(self.num_movie)))

            # Start with zero
            # for i in range(len(raw_data)):
            #     raw_data["userid"][i] = map_userid[raw_data["userid"][i]]
            #     raw_data["movieid"][i] = map_movieid[raw_data["movieid"][i]]

            # Groupby in one iteration
            movie_dict = {
                id_movie: data
                for id_movie, data in raw_data.groupby("movieid")
            }
            # for id_movie, data in raw_data.groupby("movieid"):
            #     movie_dict[id_movie] = data

            # Split into ```num_client``` sets
            idx_movie = movie_unique
            shuffle(idx_movie)
            idx_movie_split = np.array_split(idx_movie, self.num_client)

            vfl_data = collections.defaultdict(dict)
            # Fill data
            for id_subset, subset_movie in enumerate(idx_movie_split):
                id_client = id_subset + 1
                # Obtain movies belong to this subset
                data = pd.DataFrame()
                for id_movie in subset_movie:
                    data = data.append(movie_dict[id_movie])
                # Get user data
                samples = list()
                for id_user, sub_data in data.groupby("userid"):
                    choice_data = sub_data.sample(
                        n=self.theta, replace=len(sub_data) <= self.theta)
                    id_user_map = map_userid[id_user]
                    items = [
                        map_movieid[_] for _ in choice_data["movieid"].values
                    ]
                    samples.append(
                        [id_user_map, items, choice_data["rating"].values])
                # Split train/test
                train_samples, test_samples = train_test_split(samples,
                                                               train_size=0.8,
                                                               test_size=0.2,
                                                               shuffle=True)
                vfl_data[id_client]["train_data"] = train_samples
                vfl_data[id_client]["test_data"] = test_samples

            # Saving for next time
            with open(path_data, 'wb') as f:
                pickle.dump(
                    {
                        "data": vfl_data,
                        "num_user": self.num_user,
                        "num_movie": self.num_movie
                    }, f)

            self.data = vfl_data

        config.model.num_user = self.num_user
        config.model.num_item = self.num_movie

    def _download(self) -> None:
        """Download the MovieLens data if it doesn't exist in processed_folder already."""
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url,
                                         download_root=self.processed_folder,
                                         filename=filename,
                                         md5=md5)
