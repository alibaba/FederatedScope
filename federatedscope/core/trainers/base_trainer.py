import abc
import inspect


class BaseTrainer(abc.ABC):
    def __init__(self, model, data, device, **kwargs):
        self.model = model
        self.data = data
        self.device = device
        self.kwargs = kwargs

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, target_data_split_name='test'):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model_parameters, strict=False):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_para(self):
        raise NotImplementedError

    def print_trainer_meta_info(self):
        """
        Returns: String contains meta information of Trainer.
        """
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'

    def save_model(self, path, cur_round):
        raise NotImplementedError
