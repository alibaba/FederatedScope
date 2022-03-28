import collections

from torch.utils.data import DataLoader
from federatedscope.mf.dataset.movielens1m import MovieLens1M

from torch.utils.data import Dataset


def load_mf_dataset(config=None):
    """Return dataset of matrix factorization

    Format:
        {
            'client_id': {
                'train': DataLoader(),
                'test': DataLoader(),
                'val': DataLoader()
            }
        }

    """
    if config.data.type.lower() == "movielens1m":
        dataset = MovieLens1M(config,
                              root=config.data.root,
                              num_client=config.federate.client_num,
                              theta=config.sgdmf.theta)
    else:
        raise NotImplementedError("Dataset {} is not implemented.".format(
            config.data.type))

    data_local_dict = collections.defaultdict(dict)
    for id_client, data in dataset.data.items():
        train_data, test_data = data["train_data"], data["test_data"]
        data_local_dict[id_client]["train"] = DataLoader(
            train_data,
            shuffle=config.data.shuffle,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            drop_last=config.data.drop_last)
        data_local_dict[id_client]["test"] = DataLoader(
            test_data,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            drop_last=config.data.drop_last)
    return data_local_dict, config
