import abc
import inspect


class BaseSplitter(abc.ABC):
    """
    This is an abstract base class for all splitter, which is not \
    implemented with ``__call__()``.

    Attributes:
        client_num: Divide the dataset into ``client_num`` pieces.
    """
    def __init__(self, client_num):
        self.client_num = client_num

    @abc.abstractmethod
    def __call__(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        """

        Returns: Meta information for `Splitter`.

        """
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'
