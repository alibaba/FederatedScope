from federatedscope.core.data.base_translator import BaseDataTranslator
from federatedscope.core.data.base_data import ClientData


class DummyDataTranslator(BaseDataTranslator):
    """
    DummyDataTranslator convert FL dataset to DataLoader.
    Do not perform ML split and FL split.
    """
    def split(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError(f'Not support data type {type(dataset)}')
        datadict = {}
        for client_id in dataset.keys():
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            datadict[client_id] = ClientData(client_cfg, **dataset[client_id])
        return datadict
