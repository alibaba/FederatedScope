from federatedscope.vertical_fl.tree_based_models.worker import \
    wrap_client_for_train, wrap_server_for_train, \
    wrap_client_for_evaluation, wrap_server_for_evaluation
from federatedscope.vertical_fl.tree_based_models.worker.he_evaluation_wrapper\
    import wrap_client_for_he_evaluation


def wrap_vertical_server(server, config):
    if config.vertical.algo in ['xgb', 'gbdt', 'rf']:
        server = wrap_server_for_train(server)
        server = wrap_server_for_evaluation(server)

    return server


def wrap_vertical_client(client, config):
    if config.vertical.algo in ['xgb', 'gbdt', 'rf']:
        if config.vertical.eval_protection == 'he':
            client = wrap_client_for_he_evaluation(client)
        else:
            client = wrap_client_for_evaluation(client)
        client = wrap_client_for_train(client)
    return client
