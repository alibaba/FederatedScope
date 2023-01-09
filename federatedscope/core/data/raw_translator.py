from federatedscope.core.data.base_translator import BaseDataTranslator


class RawDataTranslator(BaseDataTranslator):
    def __init__(self, global_cfg, client_cfgs=None):
        self.global_cfg = global_cfg
        self.client_cfgs = client_cfgs

    def __call__(self, dataset):
        return dataset
