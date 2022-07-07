import logging

from federatedscope.core.configs import constants
from federatedscope.core.worker import Server, Client

logger = logging.getLogger(__name__)


def get_client_cls(cfg):
    if cfg.hpo.fedex.use:
        from federatedscope.autotune.fedex import FedExClient
        return FedExClient

    if cfg.vertical.use:
        from federatedscope.vertical_fl.worker import vFLClient
        return vFLClient

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
    return client_class


def get_server_cls(cfg):
    if cfg.hpo.fedex.use:
        from federatedscope.autotune.fedex import FedExServer
        return FedExServer

    if cfg.attack.attack_method.lower() in ['dlg', 'ig']:
        from federatedscope.attack.worker_as_attacker.server_attacker import\
            PassiveServer
        return PassiveServer
    elif cfg.attack.attack_method.lower() in ['passivepia']:
        from federatedscope.attack.worker_as_attacker.server_attacker import\
            PassivePIAServer
        return PassivePIAServer

    if cfg.vertical.use:
        from federatedscope.vertical_fl.worker import vFLServer
        return vFLServer

    if cfg.federate.method.lower() in constants.SERVER_TYPE:
        client_type = constants.SERVER_TYPE[cfg.federate.method.lower()]
    else:
        client_type = "normal"
        logger.warning(
            'Server for method {} is not implemented. Will use default one'.
            format(cfg.federate.method))

    if client_type == 'fedsageplus':
        from federatedscope.gfl.fedsageplus.worker import FedSagePlusServer
        return FedSagePlusServer
    elif client_type == 'gcflplus':
        from federatedscope.gfl.gcflplus.worker import GCFLPlusServer
        return GCFLPlusServer
    else:
        return Server
