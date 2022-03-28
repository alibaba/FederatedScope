import logging

import numpy as np
import torch

from federatedscope.core import constants
import federatedscope.register as register
from federatedscope.core.aggregator import *


def get_aggregator(method, model=None, device=None, online=False, config=None):
    if method.lower() in constants.AGGREGATOR_TYPE:
        aggregator_type = constants.AGGREGATOR_TYPE[method.lower()]
    else:
        aggregator_type = "clients_avg"
        logging.warning(
            'Aggregator for method {} is not implemented. Will use default one'
            .format(method))

    if aggregator_type == 'clients_avg':
        if online:
            return OnlineClientsAvgAggregator(
                model=model,
                device=device,
                src_device=device
                if config.federate.share_local_model else 'cpu')
        else:
            return ClientsAvgAggregator(model=model, device=device)
    elif aggregator_type == 'server_clients_interpolation':
        return ServerClientsInterpolateAggregator(
            model=model, device=device, beta=config.personalization.beta)
    elif aggregator_type == 'fedopt':
        return FedOptAggregator(config=config, model=model, device=device)
    else:
        raise NotImplementedError(
            "Aggregator {} is not implemented.".format(aggregator_type))
    # elif cfg.aggregator.type.lower() == 'fednova':
    #     return FedNovaAggregator()
