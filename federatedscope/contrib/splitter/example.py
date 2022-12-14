from federatedscope.register import register_splitter
from federatedscope.core.splitters import BaseSplitter


class MySplitter(BaseSplitter):
    def __init__(self, client_num, **kwargs):
        super(MySplitter, self).__init__(client_num, **kwargs)

    def __call__(self, dataset, *args, **kwargs):
        # Dummy splitter, only for demonstration
        per_samples = len(dataset) // self.client_num
        data_list, cur_index = [], 0
        for i in range(self.client_num):
            data_list.append(
                [x for x in range(cur_index, cur_index + per_samples)])
        cur_index += per_samples
        return data_list


def call_my_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'mysplitter':
        splitter = MySplitter(client_num, **kwargs)
        return splitter


register_splitter('mysplitter', call_my_splitter)
