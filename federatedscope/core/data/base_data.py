import logging

from scipy.sparse.csc import csc_matrix

from federatedscope.core.data.utils import merge_data
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader

logger = logging.getLogger(__name__)


class StandaloneDataDict(dict):
    """
    ``StandaloneDataDict`` maintain several ``ClientData``, only used in \
    ``Standalone`` mode to be passed to ``Runner``, which will conduct \
    several preprocess based on ``global_cfg``, see ``preprocess()`` \
    for details.

    Args:
        datadict: ``Dict`` with ``client_id`` as key,  ``ClientData`` as value.
        global_cfg: global ``CfgNode``
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
        Reset-up new configs for ``ClientData``, when the configs change \
        which might be used in HPO.

        Args:
            global_cfg: enable new config for ``ClientData``
            client_cfgs: enable new client-specific config for ``ClientData``
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
        Preprocess for:

        (1) Global evaluation (merge test data).
        (2) Global mode (train with centralized setting, merge all data).
        (3) Apply data attack algorithms.

        Args:
            datadict: dict with `client_id` as key,  `ClientData` as value.
        """
        if self.global_cfg.federate.merge_test_data:
            merge_split = ['test']
            if self.global_cfg.federate.merge_val_data:
                merge_split += ['val']
            server_data = merge_data(
                all_data=datadict,
                merged_max_data_id=self.global_cfg.federate.client_num,
                specified_dataset_name=merge_split)
            # `0` indicate Server
            datadict[0] = ClientData(self.global_cfg, **server_data)

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
                    merged_data = merge_data(
                        all_data=datadict,
                        merged_max_data_id=self.global_cfg.federate.client_num)
                    datadict[1] = ClientData(self.global_cfg, **merged_data)
        datadict = self.attack(datadict)
        return datadict

    def attack(self, datadict):
        """
        Apply attack to ``StandaloneDataDict``.
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
    ``ClientData`` converts split data to ``DataLoader``.

    Args:
        loader: ``Dataloader`` class or data dict which have been built
        client_cfg: client-specific ``CfgNode``
        data: raw dataset, which will stay raw
        train: train dataset, which will be converted to ``Dataloader``
        val: valid dataset, which will be converted to ``Dataloader``
        test: test dataset, which will be converted to ``Dataloader``

    Note:
        Key ``{split}_data`` in ``ClientData`` is the raw dataset.
        Key ``{split}`` in ``ClientData`` is the dataloader.
    """
    SPLIT_NAMES = ['train', 'val', 'test']

    def __init__(self, client_cfg, train=None, val=None, test=None, **kwargs):
        self.client_cfg = None
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.setup(client_cfg)
        if kwargs is not None:
            for key in kwargs:
                self[key] = kwargs[key]
        super(ClientData, self).__init__()

    def setup(self, new_client_cfg=None):
        """
        Set up ``DataLoader`` in ``ClientData`` with new configurations.

        Args:
            new_client_cfg: new client-specific CfgNode

        Returns:
            Bool: Status for indicating whether the client_cfg is updated
        """
        # if `batch_size` or `shuffle` change, re-instantiate DataLoader
        if self.client_cfg is not None:
            if dict(self.client_cfg.dataloader) == dict(
                    new_client_cfg.dataloader):
                return False

        self.client_cfg = new_client_cfg

        for split_data, split_name in zip(
            [self.train_data, self.val_data, self.test_data],
                self.SPLIT_NAMES):
            if split_data is not None:
                # csc_matrix does not have ``__len__`` attributes
                if isinstance(split_data, csc_matrix):
                    self[split_name] = get_dataloader(split_data,
                                                      self.client_cfg,
                                                      split_name)
                elif len(split_data) > 0:
                    self[split_name] = get_dataloader(split_data,
                                                      self.client_cfg,
                                                      split_name)

        return True
