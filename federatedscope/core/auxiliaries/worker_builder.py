import logging

from federatedscope.core.configs import constants
from federatedscope.core.workers import Server, Client
import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.worker import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.worker`, some modules are not '
        f'available.')


def get_client_cls(cfg):
    """
    This function return a class of client.

    Args:
        cfg: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        A client class decided by ``cfg``.

    Note:
      The key-value pairs of client type and source:
        ==================== ==============================================
        Client type          Source
        ==================== ==============================================
        ``local``            ``core.workers.Client``
        ``fedavg``           ``core.workers.Client``
        ``pfedme``           ``core.workers.Client``
        ``ditto``            ``core.workers.Client``
        ``fedex``            ``autotune.fedex.FedExClient``
        ``vfl``              ``vertical_fl.worker.vFLClient``
        ``fedsageplus``      ``gfl.fedsageplus.worker.FedSagePlusClient``
        ``gcflplus``         ``gfl.gcflplus.worker.GCFLPlusClient``
        ``gradascent``       \
        ``attack.worker_as_attacker.active_client``
        ==================== ==============================================
    """
    for func in register.worker_dict.values():
        worker_class = func(cfg.federate.method.lower())
        if worker_class is not None:
            return worker_class['client']

    if cfg.hpo.fedex.use:
        from federatedscope.autotune.fedex import FedExClient
        return FedExClient
    if cfg.hpo.fts.use:
        from federatedscope.autotune.fts import FTSClient
        return FTSClient
    if cfg.hpo.pfedhpo.use:
        from federatedscope.autotune.pfedhpo import pFedHPOClient
        return pFedHPOClient

    if cfg.vertical.use:
        if cfg.vertical.algo == 'lr':
            from federatedscope.vertical_fl.linear_model.worker \
                import vFLClient
            return vFLClient
        elif cfg.vertical.algo in ['xgb', 'gbdt', 'rf']:
            from federatedscope.vertical_fl.tree_based_models.worker \
                import TreeClient
            return TreeClient
        else:
            raise ValueError(f'No client class for {cfg.vertical.algo}')

    if cfg.data.type.lower() == 'hetero_nlp_tasks':
        from federatedscope.nlp.hetero_tasks.worker import ATCClient
        return ATCClient

    if cfg.federate.method.lower() in constants.CLIENTS_TYPE:
        client_type = constants.CLIENTS_TYPE[cfg.federate.method.lower()]
    else:
        client_type = "normal"
        logger.warning(
            'Clients for method {} is not implemented. Will use default one'.
            format(cfg.federate.method))

    if client_type == 'fedsageplus':
        from federatedscope.gfl.fedsageplus.worker import FedSagePlusClient
        client_class = FedSagePlusClient
    elif client_type == 'gcflplus':
        from federatedscope.gfl.gcflplus.worker import GCFLPlusClient
        client_class = GCFLPlusClient
    elif client_type == 'fedgc':
        from federatedscope.cl.fedgc.client import GlobalContrastFLClient
        client_class = GlobalContrastFLClient
    else:
        client_class = Client

    # add attack related method to client_class

    if cfg.attack.attack_method.lower() in constants.CLIENTS_TYPE:
        client_atk_type = constants.CLIENTS_TYPE[
            cfg.attack.attack_method.lower()]
    else:
        client_atk_type = None

    if client_atk_type == 'gradascent':
        from federatedscope.attack.worker_as_attacker.active_client import \
            add_atk_method_to_Client_GradAscent
        logger.info("=========== add method to current client class ")
        client_class = add_atk_method_to_Client_GradAscent(client_class)

    if cfg.llm.offsite_tuning.use:
        from federatedscope.llm.offsite_tuning.client import \
            OffsiteTuningClient
        logger.info("=========== Using offsite_tuning ===========")
        return OffsiteTuningClient

    return client_class


def get_server_cls(cfg):
    """
    This function return a class of server.

    Args:
        cfg: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        A server class decided by ``cfg``.

    Note:
      The key-value pairs of server type and source:
        ==================== ==============================================
        Server type          Source
        ==================== ==============================================
        ``local``            ``core.workers.Server``
        ``fedavg``           ``core.workers.Server``
        ``pfedme``           ``core.workers.Server``
        ``ditto``            ``core.workers.Server``
        ``fedex``            ``autotune.fedex.FedExServer``
        ``vfl``              ``vertical_fl.worker.vFLServer``
        ``fedsageplus``      ``gfl.fedsageplus.worker.FedSagePlusServer``
        ``gcflplus``         ``gfl.gcflplus.worker.GCFLPlusServer``
        ``attack``           \
        ``attack.worker_as_attacker.server_attacker.PassiveServer`` and \
        ``attack.worker_as_attacker.server_attacker.PassivePIAServer``
        ``backdoor``         \
        ``attack.worker_as_attacker.server_attacker.BackdoorServer``
        ==================== ==============================================
    """
    for func in register.worker_dict.values():
        worker_class = func(cfg.federate.method.lower())
        if worker_class is not None:
            return worker_class['server']

    if cfg.hpo.fedex.use:
        from federatedscope.autotune.fedex import FedExServer
        return FedExServer

    if cfg.hpo.fts.use:
        from federatedscope.autotune.fts import FTSServer
        return FTSServer
    if cfg.hpo.pfedhpo.use and not cfg.hpo.pfedhpo.train_fl:
        from federatedscope.autotune.pfedhpo import pFedHPOServer
        return pFedHPOServer
    if cfg.hpo.pfedhpo.use and cfg.hpo.pfedhpo.train_fl:
        from federatedscope.autotune.pfedhpo import pFedHPOFLServer
        return pFedHPOFLServer

    if cfg.attack.attack_method.lower() in ['dlg', 'ig']:
        from federatedscope.attack.worker_as_attacker.server_attacker import\
            PassiveServer
        return PassiveServer
    elif cfg.attack.attack_method.lower() in ['passivepia']:
        from federatedscope.attack.worker_as_attacker.server_attacker import\
            PassivePIAServer
        return PassivePIAServer

    elif cfg.attack.attack_method.lower() in ['backdoor']:
        from federatedscope.attack.worker_as_attacker.server_attacker \
            import BackdoorServer
        return BackdoorServer

    if cfg.vertical.use:
        if cfg.vertical.algo == 'lr':
            from federatedscope.vertical_fl.linear_model.worker \
                import vFLServer
            return vFLServer
        elif cfg.vertical.algo in ['xgb', 'gbdt', 'rf']:
            from federatedscope.vertical_fl.tree_based_models.worker \
                import TreeServer
            return TreeServer
        else:
            raise ValueError(f'No server class for {cfg.vertical.algo}')

    if cfg.data.type.lower() == 'hetero_nlp_tasks':
        from federatedscope.nlp.hetero_tasks.worker import ATCServer
        return ATCServer

    if cfg.federate.method.lower() in constants.SERVER_TYPE:
        server_type = constants.SERVER_TYPE[cfg.federate.method.lower()]
    else:
        server_type = "normal"
        logger.warning(
            'Server for method {} is not implemented. Will use default one'.
            format(cfg.federate.method))

    if server_type == 'fedsageplus':
        from federatedscope.gfl.fedsageplus.worker import FedSagePlusServer
        server_class = FedSagePlusServer
    elif server_type == 'gcflplus':
        from federatedscope.gfl.gcflplus.worker import GCFLPlusServer
        server_class = GCFLPlusServer
    elif server_type == 'fedgc':
        from federatedscope.cl.fedgc.server import GlobalContrastFLServer
        server_class = GlobalContrastFLServer
    else:
        server_class = Server

    if cfg.llm.offsite_tuning.use:
        from federatedscope.llm.offsite_tuning.server import \
            OffsiteTuningServer
        logger.info("=========== Using offsite_tuning ===========")
        return OffsiteTuningServer

    return server_class
