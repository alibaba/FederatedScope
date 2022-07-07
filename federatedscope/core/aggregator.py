from abc import ABC, abstractmethod
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer

import torch
import os


class Aggregator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        pass


class ClientsAvgAggregator(Aggregator):
    """Implementation of vanilla FedAvg refer to `Communication-efficient
    learning of deep networks from decentralized data` [McMahan et al., 2017]
        (http://proceedings.mlr.press/v54/mcmahan17a.html)
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

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

    def _para_weighted_avg(self, models, recover_fun=None):
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        for key in avg_model:
            for i in range(len(models)):
                local_sample_size, local_model = models[i]

                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif self.cfg.federate.use_ss:
                    # When using secret sharing, what the server receives
                    # are sample_size * model_para
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are
                # sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model


class NoCommunicationAggregator(Aggregator):
    """"Clients do not communicate. Each client work locally
    """
    def aggregate(self, agg_info):
        # do nothing
        return {}


class OnlineClientsAvgAggregator(ClientsAvgAggregator):
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineClientsAvgAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        return self.maintained


class ServerClientsInterpolateAggregator(ClientsAvgAggregator):
    """"
        # conduct aggregation by interpolating global model from server and
        local models from clients
    """
    def __init__(self, model=None, device='cpu', config=None, beta=1.0):
        super(ServerClientsInterpolateAggregator,
              self).__init__(model, device, config)
        self.beta = beta  # the weight for local models used in interpolation

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        global_model = self.model
        elem_each_client = next(iter(models))
        assert len(elem_each_client) == 2, f"Require (sample_size, " \
                                           f"model_para) tuple for each " \
                                           f"client, i.e., len=2, but got " \
                                           f"len={len(elem_each_client)}"
        avg_model_by_clients = self._para_weighted_avg(models)
        global_local_models = [((1 - self.beta), global_model.state_dict()),
                               (self.beta, avg_model_by_clients)]

        avg_model_by_interpolate = self._para_weighted_avg(global_local_models)
        return avg_model_by_interpolate


class FedOptAggregator(ClientsAvgAggregator):
    """Implementation of FedOpt refer to `Adaptive Federated Optimization` [
    Reddi et al., 2021]
        (https://openreview.net/forum?id=LkFG3lB13U5)

    """
    def __init__(self, config, model, device='cpu'):
        super(FedOptAggregator, self).__init__(model, device, config)
        self.optimizer = get_optimizer(model=self.model,
                                       **config.fedopt.optimizer)

    def aggregate(self, agg_info):
        new_model = super().aggregate(agg_info)

        model = self.model.cpu().state_dict()
        with torch.no_grad():
            grads = {key: model[key] - new_model[key] for key in new_model}

        self.optimizer.zero_grad()
        for key, p in self.model.named_parameters():
            if key in new_model.keys():
                p.grad = grads[key]
        self.optimizer.step()

        return self.model.state_dict()
