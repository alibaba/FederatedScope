import logging
from federatedscope.core.configs import constants

logger = logging.getLogger(__name__)


def get_aggregator(method, model=None, device=None, online=False, config=None):
    """
    This function builds an aggregator, which is a protocol for aggregate \
    all clients' model(s).

    Arguments:
        method: key to determine which aggregator to use
        model:  model to be aggregated
        device: where to aggregate models (``cpu`` or ``gpu``)
        online: ``True`` or ``False`` to use online aggregator.
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        An instance of aggregator (see ``core.aggregator`` for details)

    Note:
      The key-value pairs of ``method`` and aggregators:
        ==================================  ===========================
        Method                              Aggregator
        ==================================  ===========================
        ``tensorflow``                      ``cross_backends.FedAvgAggregator``
        ``local``                           \
        ``core.aggregators.NoCommunicationAggregator``
        ``global``                          \
        ``core.aggregators.NoCommunicationAggregator``
        ``fedavg``                          \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``pfedme``                          \
        ``core.aggregators.ServerClientsInterpolateAggregator``
        ``ditto``                           \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``fedsageplus``                     \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``gcflplus``                        \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``fedopt``                          \
        ``core.aggregators.FedOptAggregator``
        ==================================  ===========================
    """
    if config.backend == 'tensorflow':
        from federatedscope.cross_backends import FedAvgAggregator
        return FedAvgAggregator(model=model, device=device)
    else:
        from federatedscope.core.aggregators import ClientsAvgAggregator, \
            OnlineClientsAvgAggregator, ServerClientsInterpolateAggregator, \
            FedOptAggregator, NoCommunicationAggregator, \
            AsynClientsAvgAggregator, KrumAggregator, \
            MedianAggregator, TrimmedmeanAggregator, \
            BulyanAggregator,  NormboundingAggregator

    STR2AGG = {
        'fedavg': ClientsAvgAggregator,
        'krum': KrumAggregator,
        'median': MedianAggregator,
        'bulyan': BulyanAggregator,
        'trimmedmean': TrimmedmeanAggregator,
        'normbounding': NormboundingAggregator
    }

    if method.lower() in constants.AGGREGATOR_TYPE:
        aggregator_type = constants.AGGREGATOR_TYPE[method.lower()]
    else:
        aggregator_type = "clients_avg"
        logger.warning(
            'Aggregator for method {} is not implemented. Will use default one'
            .format(method))

    if config.data.type.lower() == 'hetero_nlp_tasks' and \
            not config.federate.atc_vanilla:
        from federatedscope.nlp.hetero_tasks.aggregator import ATCAggregator
        return ATCAggregator(model=model, config=config, device=device)

    if config.fedopt.use or aggregator_type == 'fedopt':
        return FedOptAggregator(config=config, model=model, device=device)
    elif aggregator_type == 'clients_avg':
        if online:
            return OnlineClientsAvgAggregator(
                model=model,
                device=device,
                config=config,
                src_device=device
                if config.federate.share_local_model else 'cpu')
        elif config.asyn.use:
            return AsynClientsAvgAggregator(model=model,
                                            device=device,
                                            config=config)
        else:
            if config.aggregator.robust_rule not in STR2AGG:
                logger.warning(
                    f'The specified {config.aggregator.robust_rule} aggregtion\
                    rule has not been supported, the vanilla fedavg algorithm \
                    will be used instead.')
            return STR2AGG.get(config.aggregator.robust_rule,
                               ClientsAvgAggregator)(model=model,
                                                     device=device,
                                                     config=config)

    elif aggregator_type == 'server_clients_interpolation':
        return ServerClientsInterpolateAggregator(
            model=model,
            device=device,
            config=config,
            beta=config.personalization.beta)
    elif aggregator_type == 'no_communication':
        return NoCommunicationAggregator(model=model,
                                         device=device,
                                         config=config)
    else:
        raise NotImplementedError(
            "Aggregator {} is not implemented.".format(aggregator_type))
