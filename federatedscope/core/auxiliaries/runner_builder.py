from federatedscope.core.fed_runner import StandaloneRunner, DistributedRunner


def get_runner(data, server_class, client_class, config, client_configs=None):
    # Instantiate a Runner based on a configuration file
    mode = config.federate.mode.lower()
    runner_dict = {
        'standalone': StandaloneRunner,
        'distributed': DistributedRunner
    }
    return runner_dict[mode](data=data,
                             server_class=server_class,
                             client_class=client_class,
                             config=config,
                             client_configs=client_configs)
