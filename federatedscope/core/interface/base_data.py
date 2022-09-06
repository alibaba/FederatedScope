import logging
from federatedscope.core.auxiliaries.utils import merge_data

logger = logging.getLogger(__name__)


class StandaloneDataDict(dict):
    """
        `StandaloneDataDict` maintain several `ClientData`.
    """
    client_cfgs = None

    def __init__(self, datadict, global_cfg):
        """

        Args:
            datadict: `Dict` with `client_id` as key,  `ClientData` as value.
            global_cfg: global CfgNode
        """
        self.cfg = global_cfg
        datadict = self.preprocess(datadict)
        super(StandaloneDataDict, self).__init__(datadict)

    def resetup(self, global_cfg, client_cfgs=None):
        """
        Resetup new configs for `ClientData`, which might be used in HPO.

        Args:
            global_cfg: enable new config for `ClientData`
            client_cfg: enable new client-specific config for `ClientData`
        """
        self.cfg, self.client_cfgs = global_cfg, client_cfgs
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
        Preprocess for StandaloneDataDict.

        Args:
            datadict: dict with `client_id` as key,  `ClientData` as value.
        """
        if self.cfg.federate.merge_test_data:
            server_data = merge_data(
                all_data=datadict,
                merged_max_data_id=self.cfg.federate.client_num,
                specified_dataset_name=['test'])
            # `0` indicate Server
            datadict[0] = server_data

        if self.cfg.federate.method == "global":
            if self.cfg.federate.client_num != 1:
                if self.cfg.data.server_holds_all:
                    assert datadict[0] is not None \
                        and len(datadict[0]) != 0, \
                        "You specified cfg.data.server_holds_all=True " \
                        "but data[0] is None. Please check whether you " \
                        "pre-process the data[0] correctly"
                    datadict[1] = datadict[0]
                else:
                    logger.info(f"Will merge data from clients whose ids in "
                                f"[1, {self.cfg.federate.client_num}]")
                    datadict[1] = merge_data(
                        all_data=datadict,
                        merged_max_data_id=self.cfg.federate.client_num)
        return datadict


class ClientData(dict):
    """
        `ClientData` converts dataset to loader.
    """
    client_cfg = None

    def __init__(self, loader, client_cfg, train=None, val=None, test=None):
        """

        Args:
            loader: Dataloader class or data dict which have been built
            client_cfg: client-specific CfgNode
            train: train dataset
            val: valid dataset
            test: test dataset
        """
        self.train = train
        self.val = val
        self.test = test
        self.loader = loader
        self.setup(client_cfg)
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
            if self.client_cfg.data.batch_size == \
                    new_client_cfg.data.batch_size or \
                    self.client_cfg.data.shuffle == \
                    new_client_cfg.data.shuffle:
                return False

        self.client_cfg = new_client_cfg
        if self.train is not None:
            self['train'] = self.loader(
                self.train,
                batch_size=new_client_cfg.data.batch_size,
                shuffle=new_client_cfg.data.shuffle,
                num_workers=new_client_cfg.data.num_workers)

        if self.val is not None:
            self['val'] = self.loader(
                self.val,
                batch_size=new_client_cfg.data.batch_size,
                shuffle=False,
                num_workers=new_client_cfg.data.num_workers)

        if self.test is not None:
            self['test'] = self.loader(
                self.test,
                batch_size=new_client_cfg.data.batch_size,
                shuffle=False,
                num_workers=new_client_cfg.data.num_workers)
        return True
