import logging

from federatedscope.core import constants
from federatedscope.core.worker import Server, Client


def get_client_cls(cfg):
    if cfg.vertical.use:
        from federatedscope.vertical_fl.worker import vFLClient
        return vFLClient

    if cfg.federate.method.lower() in constants.CLIENTS_TYPE:
        client_type = constants.CLIENTS_TYPE[cfg.federate.method.lower()]
    else:
        client_type = "normal"
        logging.warning(
            'Clients for method {} is not implemented. Will use default one'.
            format(cfg.federate.method))

    if client_type == 'fedsageplus':
        from federatedscope.gfl.fedsageplus.worker import FedSagePlusClient
        return FedSagePlusClient
    elif client_type == 'gcflplus':
        from federatedscope.gfl.gcflplus.worker import GCFLPlusClient
        return GCFLPlusClient
    else:
        return Client


def get_server_cls(cfg):
    if cfg.attack.attack_method.lower() in ['dlg', 'ig']:
        from federatedscope.attack.worker_as_attacker.server_attacker import PassiveServer
        return PassiveServer

    if cfg.vertical.use:
        from federatedscope.vertical_fl.worker import vFLServer
        return vFLServer

    if cfg.federate.method.lower() in constants.SERVER_TYPE:
        client_type = constants.SERVER_TYPE[cfg.federate.method.lower()]
    else:
        client_type = "normal"
        logging.warning(
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