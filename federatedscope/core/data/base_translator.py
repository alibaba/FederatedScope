from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.data import ClientData, StandaloneDataDict


class BaseDataTranslator:
    def __init__(self, global_cfg, loader, client_cfgs=None):
        """
        Convert data to `StandaloneDataDict`.

        Args:
            global_cfg: global CfgNode
            loader: `torch.utils.data.DataLoader` or subclass of it
            client_cfgs: client cfg `Dict`
        """
        self.loader = loader
        self.global_cfg = global_cfg.clone()
        self.client_cfgs = client_cfgs
        self.splitter = get_splitter(global_cfg)

    def __call__(self, dataset):
        """

        Args:
            dataset: `torch.utils.data.Dataset`, `List` of (feature, label)
                or split dataset tuple of (train, val, test) or Tuple of
                split dataset with [train, val, test]
        """
        train, val, test = self.split_train_val_test()
        datadict = self.split_to_client(train, val, test)
        return StandaloneDataDict(datadict, self.global_cfg)

    def split_train_val_test(self, dataset):
        """
        Split dataset to train, val, test if not provided.

        Returns:
            split_data (List): List of split dataset, [train, val, test]

        """
        splits = self.global_cfg.data.splits
        if isinstance(dataset, tuple):
            # No need to split train/val/test for tuple dataset.
            error_msg = 'If dataset is tuple, it must contains ' \
                        'train, valid and test split.'
            assert len(dataset) == len(['train', 'val', 'test']), error_msg
            return [dataset[0], dataset[1], dataset[2]]

        from torch.utils.data.dataset import random_split
        train_size = int(splits[0] * len(dataset))
        val_size = int(splits[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        split_data = random_split(dataset, [train_size, val_size, test_size])
        return split_data

    def split_to_client(self, train, val, test):
        """
        Split dataset to clients.

        Returns:
            datadict (dict): dict of `ClientData` with client_idx as key.

        """

        # Initialization
        client_num = self.global_cfg.federate.client_num
        split_train, split_val, split_test = [[None] * client_num] * 3
        train_label_distribution = None

        # Split train/val/test to client
        if len(train) > 0:
            split_train = self.splitter(train)
            train_label_distribution = [[j[1] for j in x] for x in split_train]
        if len(val) > 0:
            split_val = self.splitter(val, prior=train_label_distribution)
        if len(test) > 0:
            split_test = self.splitter(test, prior=train_label_distribution)

        # Build data dict with `ClientData`
        datadict = {}
        for client_id in range(1, client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            datadict[client_id] = ClientData(self.loader,
                                             client_cfg,
                                             train=split_train[client_id - 1],
                                             val=split_val[client_id - 1],
                                             test=split_test[client_id - 1])
        return datadict


class DummyDataTranslator(BaseDataTranslator):
    """
    Translator split data_dict to `StandaloneDataDict`.
    """
    def __call__(self, datadict):
        if not isinstance(datadict, StandaloneDataDict):
            datadict = StandaloneDataDict(datadict, self.global_cfg)
        return datadict
