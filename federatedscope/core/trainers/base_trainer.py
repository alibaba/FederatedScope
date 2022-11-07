import abc


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

    @abc.abstractmethod
    def print_trainer_meta_info(self):
        raise NotImplementedError
