from federatedscope.core.data.base_translator import BaseDataTranslator
from federatedscope.core.data.base_data import ClientData


class DummyDataTranslator(BaseDataTranslator):
    """
    ``DummyDataTranslator`` convert datadict to ``StandaloneDataDict``. \
    Compared to ``core.data.base_translator.BaseDataTranslator``, it do not \
    perform FL split.
    """
    def split(self, dataset):
        """
        Perform ML split

        Returns:
            dict of ``ClientData`` with client_idx as key to build \
            ``StandaloneDataDict``
        """
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

            if isinstance(dataset[client_id], dict):
                datadict[client_id] = ClientData(client_cfg,
                                                 **dataset[client_id])
            else:
                # Do not have train/val/test
                train, val, test = self.split_train_val_test(
                    dataset[client_id], client_cfg)
                tmp_dict = dict(train=train, val=val, test=test)
                # Only for graph-level task, get number of graph labels
                if client_cfg.model.task.startswith('graph') and \
                        client_cfg.model.out_channels == 0:
                    s = set()
                    for g in dataset[client_id]:
                        s.add(g.y.item())
                    tmp_dict['num_label'] = len(s)
                datadict[client_id] = ClientData(client_cfg, **tmp_dict)
        return datadict
