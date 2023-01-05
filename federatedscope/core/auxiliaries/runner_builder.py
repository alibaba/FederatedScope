from federatedscope.core.fed_runner import StandaloneRunner, DistributedRunner, StandaloneMultiProcessRunner
from federatedscope.core.auxiliaries.parallel_runner import StandaloneMultiGPURunner
from federatedscope.core.auxiliaries.data_builder import get_data


def get_runner(server_class, client_class, config, client_configs=None):
    """
    Instantiate a runner based on a configuration file

    Args:
        server_class: server class
        client_class: client class
        config: configurations for FL, see ``federatedscope.core.configs``
        client_configs: client-specific configurations

    Returns:
        An instantiated FedRunner to run the FL course.

    Note:
      The key-value pairs of built-in runner and source are shown below:
        ============================================  ===============================
        Mode                                          Source
        ============================================  ===============================
        ``standalone``                                ``core.fed_runner.StandaloneRunner``
        ``distributed``                               ``core.fed_runner.DistributedRunner``
        ``standalone(multi_gpu=False,prcess_num>1)``  ``core.fed_runner.StandaloneMultiProcessRunner``
        ``standalone(multi_gpu=True,process_num>1)``  ``core.fed_runner.StandaloneMultiGPURunner``
        ============================================  ===============================
    """

    mode = config.federate.mode.lower()
    multi_gpu = config.federate.multi_gpu
    process_num = config.federate.process_num

    if mode == 'standalone':
        if process_num == 1:
            runner_cls = StandaloneRunner
        elif multi_gpu:
            runner_cls = StandaloneMultiGPURunner
        else:
            runner_cls = StandaloneMultiProcessRunner
    elif mode == 'distributed':
        runner_cls = DistributedRunner
    
    data = None
    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    if runner_cls is not StandaloneMultiGPURunner:
        data, modified_cfg = get_data(config=config.clone(),
                                    client_cfgs=client_configs)
        config.merge_from_other_cfg(modified_cfg)
        config.freeze()

    return runner_cls(data=data,
                      server_class=server_class,
                      client_class=client_class,
                      config=config,
                      client_configs=client_configs)
