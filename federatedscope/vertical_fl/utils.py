from federatedscope.vertical_fl.tree_based_models.worker import \
    wrap_client_for_train, wrap_server_for_train, \
    wrap_client_for_evaluation, wrap_server_for_evaluation
from federatedscope.vertical_fl.tree_based_models.worker.he_evaluation_wrapper\
    import wrap_client_for_he_evaluation
from federatedscope.vertical_fl.tree_based_models.worker.ss_evaluation_wrapper\
    import wrap_client_for_ss_evaluation, wrap_server_for_ss_evaluation
from federatedscope.core.secret_sharing.ss_multiplicative_wrapper import \
    wrap_server_for_ss_multiplicative, wrap_client_for_ss_multiplicative


def wrap_vertical_server(server, config):
    if config.vertical.algo in ['xgb', 'gbdt', 'rf']:
        server = wrap_server_for_train(server)
        server = wrap_server_for_evaluation(server)
        if config.vertical.eval_protection == 'ss':
            server = wrap_server_for_ss_evaluation(server)
            server = wrap_server_for_ss_multiplicative(server)

    return server


def wrap_vertical_client(client, config):
    if config.vertical.algo in ['xgb', 'gbdt', 'rf']:
        if config.vertical.eval_protection == 'he':
            client = wrap_client_for_he_evaluation(client)
        elif config.vertical.eval_protection == 'ss':
            client = wrap_client_for_ss_evaluation(client)
            client = wrap_client_for_ss_multiplicative(client)
        else:
            client = wrap_client_for_evaluation(client)
        client = wrap_client_for_train(client)
    return client
