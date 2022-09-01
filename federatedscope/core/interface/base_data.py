import logging
from federatedscope.core.auxiliaries.utils import merge_data

logger = logging.getLogger(__name__)


class StandaloneDataDict(dict):
    """
        `StandaloneDataDict` maintain several `ClientData`.
    """
    def __init__(self, datadict, global_cfg):
        """

        Args:
            datadict: dict with `client_id` as key,  `ClientData` as value.
            global_cfg: global CfgNode
        """
        self.cfg = global_cfg
        datadict = self.preprocess(datadict)
        super(StandaloneDataDict, self).__init__(datadict)

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
            loader: Dataloader class or dict
            client_cfg: client-specific CfgNode
            train: train dataset
            val: valid dataset
            test: test dataset
        """
        if isinstance(loader, dict):
            super(ClientData, self).__init__(loader)
        else:
            self.train = train
            self.val = val
            self.test = test
            self.loader = loader
            self._setup(client_cfg)
            super(ClientData, self).__init__()

    def _setup(self, new_client_cfg=None):
        """

        Args:
            new_client_cfg: new client-specific CfgNode

        Returns:
            Status: indicate whether the client_cfg is updated
        """
        if new_client_cfg == self.client_cfg:
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
