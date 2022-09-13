import abc
import inspect


class BaseSplitter(abc.ABC):
    def __init__(self, client_num, **kwargs):
        self.client_num = client_num
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'
