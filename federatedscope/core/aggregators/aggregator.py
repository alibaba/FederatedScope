import os
import torch
from abc import ABC, abstractmethod


class Aggregator(ABC):
    """
    Abstract class of Aggregator.
    """
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            agg_info: information to be aggregated.
        """
        pass


class NoCommunicationAggregator(Aggregator):
    """Clients do not communicate. Each client work locally
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def update(self, model_parameters):
        '''
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def aggregate(self, agg_info):
        """
        Aggregation function.

        Args:
            agg_info: information to be aggregated.
        """
        # do nothing
        return {}
