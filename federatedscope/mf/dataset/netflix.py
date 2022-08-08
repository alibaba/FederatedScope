import os
import tarfile
import logging

import pandas as pd
import numpy as np

from federatedscope.mf.dataset import MovieLensData, HMFDataset, VMFDataset

logger = logging.getLogger(__name__)


class Netflix(MovieLensData):
    """Netflix Prize Dataset
        (https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz)

        Netflix Prize consists of approximately 100,000,000 ratings from
        480,189 users for 17,770 movies. Each rating in the training dataset
        consists of four entries: user, movie, rating date, and rating.
        Users and movies are represented by integer IDs, while ratings range
        from 1 to 5.
    """
    base_folder = 'Netflix'
    url = 'https://archive.org/download/nf_prize_dataset.tar' \
          '/nf_prize_dataset.tar.gz'
    filename = 'download'
    zip_md5 = 'a8f23d2d76461211c6b4c0ca6df2547d'
    raw_file = 'training_set.tar'
    raw_file_md5 = '0098ee8997ffda361a59bc0dd1bdad8b'
    mv_names = [f'mv_{str(x).rjust(7, "0")}.txt' for x in range(1, 17771)]

    def _extract_raw_file(self, dir_path):
        # Extract flag
        flag = False
        if not os.path.exists(dir_path):
            flag = True
        else:
            for name in self.mv_names:
                if not os.path.exists(os.path.join(dir_path, name)):
                    flag = True
                    break
        if flag:
            tar = tarfile.open(
                os.path.join(self.root, self.base_folder, self.filename,
                             self.raw_file))
            tar.extractall(
                os.path.join(self.root, self.base_folder, self.filename))
            tar.close()
        return

    def _read_raw(self):
        dir_path = os.path.join(self.root, self.base_folder, self.filename,
                                'training_set')
        self._extract_raw_file(dir_path)
        frames = []
        for idx, name in enumerate(self.mv_names):
            mv_id = np.int32(idx + 1)
            df = pd.read_csv(os.path.join(dir_path, name),
                             usecols=[0, 1, 2],
                             names=["userId", "rating", "date"],
                             dtype={
                                 "userId": np.int32,
                                 "movieId": np.int32,
                                 "rating": np.float32,
                                 "date": str
                             },
                             skiprows=1)
            df["movieId"] = [mv_id] * len(df)
            frames.append(df)
        data = pd.concat(frames)
        return data


class VFLNetflix(Netflix, VMFDataset):
    """Netflix dataset in HFL setting

    """
    pass


class HFLNetflix(Netflix, HMFDataset):
    """Netflix dataset in HFL setting

    """
    pass
