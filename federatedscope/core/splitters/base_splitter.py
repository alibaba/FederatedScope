import abc


class BaseSplitter(abc.ABC):
    def __init__(self, client_num, **kwargs):
        self.client_num = client_num
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        meta_info = f'client_num {self.client_num}, kwargs {self.kwargs}'
        return f'{self.__class__.__name__}({meta_info})'
