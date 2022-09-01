from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.core.interface.base_data import ClientData, \
    StandaloneDataDict

# TODO: Implement CVDataTranslator, NLPDataTranslator from `databuilder`


class DataTranslator(StandaloneDataDict):
    def __init__(self, dataset, global_cfg, package=None):
        self.splitter = get_splitter(global_cfg)
        if package is not None:
            self.transform_funcs = get_transform(global_cfg, package)
        self.process()
        super(DataTranslator, self).__init__(dataset, global_cfg)

    @staticmethod
    def process(self):
        raise NotImplementedError
