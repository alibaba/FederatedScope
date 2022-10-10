import logging
from federatedscope.core.data.utils import merge_data
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader

logger = logging.getLogger(__name__)


class StandaloneDataDict(dict):
    """
        `StandaloneDataDict` maintain several `ClientData`.
    """
    def __init__(self, datadict, global_cfg):
        """

        Args:
            datadict: `Dict` with `client_id` as key,  `ClientData` as value.
            global_cfg: global CfgNode
        """
        self.global_cfg = global_cfg
        self.client_cfgs = None
        datadict = self.preprocess(datadict)
        super(StandaloneDataDict, self).__init__(datadict)

    def resetup(self, global_cfg, client_cfgs=None):
        """
        Resetup new configs for `ClientData`, which might be used in HPO.

        Args:
            global_cfg: enable new config for `ClientData`
            client_cfgs: enable new client-specific config for `ClientData`
        """
        self.global_cfg, self.client_cfgs = global_cfg, client_cfgs
        for client_id, client_data in self.items():
            if isinstance(client_data, ClientData):
                if client_cfgs is not None:
                    client_cfg = global_cfg.clone()
                    client_cfg.merge_from_other_cfg(
                        client_cfgs.get(f'client_{client_id}'))
                else:
                    client_cfg = global_cfg
                client_data.setup(client_cfg)
            else:
                logger.warning('`client_data` is not subclass of '
                               '`ClientData`, and cannot re-setup '
                               'DataLoader with new configs.')

    def preprocess(self, datadict):
        """
        Preprocess for StandaloneDataDict for:
            1. Global evaluation (merge test data).
            2. Global mode (train with centralized setting, merge all data).

        Args:
            datadict: dict with `client_id` as key,  `ClientData` as value.
        """
        if self.global_cfg.federate.merge_test_data:
            server_data = merge_data(
                all_data=datadict,
                merged_max_data_id=self.global_cfg.federate.client_num,
                specified_dataset_name=['test'])
            # `0` indicate Server
            datadict[0] = server_data

        if self.global_cfg.federate.method == "global":
            if self.global_cfg.federate.client_num != 1:
                if self.global_cfg.data.server_holds_all:
                    assert datadict[0] is not None \
                        and len(datadict[0]) != 0, \
                        "You specified cfg.data.server_holds_all=True " \
                        "but data[0] is None. Please check whether you " \
                        "pre-process the data[0] correctly"
                    datadict[1] = datadict[0]
                else:
                    logger.info(f"Will merge data from clients whose ids in "
                                f"[1, {self.global_cfg.federate.client_num}]")
                    datadict[1] = merge_data(
                        all_data=datadict,
                        merged_max_data_id=self.global_cfg.federate.client_num)
        datadict = self.attack(datadict)
        return datadict

    def attack(self, datadict):
        """
        Apply attack to `StandaloneDataDict`.

        """
        if 'backdoor' in self.global_cfg.attack.attack_method and 'edge' in \
                self.global_cfg.attack.trigger_type:
            import os
            import torch
            from federatedscope.attack.auxiliary import \
                create_ardis_poisoned_dataset, create_ardis_test_dataset
            if not os.path.exists(self.global_cfg.attack.edge_path):
                os.makedirs(self.global_cfg.attack.edge_path)
                poisoned_edgeset = create_ardis_poisoned_dataset(
                    data_path=self.global_cfg.attack.edge_path)

                ardis_test_dataset = create_ardis_test_dataset(
                    self.global_cfg.attack.edge_path)

                logger.info("Writing poison_data to: {}".format(
                    self.global_cfg.attack.edge_path))

                with open(
                        self.global_cfg.attack.edge_path +
                        "poisoned_edgeset_training", "wb") as saved_data_file:
                    torch.save(poisoned_edgeset, saved_data_file)

                with open(
                        self.global_cfg.attack.edge_path +
                        "ardis_test_dataset.pt", "wb") as ardis_data_file:
                    torch.save(ardis_test_dataset, ardis_data_file)
                logger.warning(
                    'please notice: downloading the poisoned dataset \
                    on cifar-10 from \
                        https://github.com/ksreenivasan/OOD_Federated_Learning'
                )

        if 'backdoor' in self.global_cfg.attack.attack_method:
            from federatedscope.attack.auxiliary import poisoning
            poisoning(datadict, self.global_cfg)
        return datadict


class ClientData(dict):
    """
        `ClientData` converts dataset to train/val/test DataLoader.
        Key `data` in `ClientData` is the raw dataset.
    """
    def __init__(self, client_cfg, train=None, val=None, test=None, **kwargs):
        """

        Args:
            loader: Dataloader class or data dict which have been built
            client_cfg: client-specific CfgNode
            data: raw dataset, which will stay raw
            train: train dataset, which will be converted to DataLoader
            val: valid dataset, which will be converted to DataLoader
            test: test dataset, which will be converted to DataLoader
        """
        self.client_cfg = None
        self.train = train
        self.val = val
        self.test = test
        self.setup(client_cfg)
        if kwargs is not None:
            for key in kwargs:
                self[key] = kwargs[key]
        super(ClientData, self).__init__()

    def setup(self, new_client_cfg=None):
        """

        Args:
            new_client_cfg: new client-specific CfgNode

        Returns:
            Status: indicate whether the client_cfg is updated
        """
        # if `batch_size` or `shuffle` change, reinstantiate DataLoader
        if self.client_cfg is not None:
            if dict(self.client_cfg.dataloader) == dict(
                    new_client_cfg.dataloader):
                return False

        self.client_cfg = new_client_cfg
        if self.train is not None:
            self['train'] = get_dataloader(self.train, self.client_cfg,
                                           'train')
        if self.val is not None:
            self['val'] = get_dataloader(self.val, self.client_cfg, 'val')
        if self.test is not None:
            self['test'] = get_dataloader(self.test, self.client_cfg, 'test')
        return True
