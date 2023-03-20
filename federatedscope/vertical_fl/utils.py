from federatedscope.vertical_fl.xgb_base.worker import wrap_client_for_train, \
    wrap_server_for_train, wrap_client_for_evaluation, \
    wrap_server_for_evaluation
from federatedscope.vertical_fl.xgb_base.worker.homo_evaluation_wrapper\
    import wrap_client_for_homo_evaluation


def wrap_vertical_server(server, config):
    if config.vertical.algo in ['xgb', 'gbdt', 'rf']:
        server = wrap_server_for_train(server)
        server = wrap_server_for_evaluation(server)

    return server


def wrap_vertical_client(client, config):
    if config.vertical.algo in ['xgb', 'gbdt', 'rf']:
        if config.vertical.eval == 'homo':
            client = wrap_client_for_homo_evaluation(client)
        else:
            client = wrap_client_for_evaluation(client)
        client = wrap_client_for_train(client)
    return client
