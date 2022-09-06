from federatedscope.register import register_data


def MyData(config, client_cfgs):
    r"""
    Returns:
            data:
                {
                    '{client_id}': {
                        'train': Dataset or DataLoader,
                        'test': Dataset or DataLoader,
                        'val': Dataset or DataLoader
                    }
                }
            config:
                cfg_node
    """
    data = None
    config = config
    client_cfgs = client_cfgs
    return data, config


def call_my_data(config, client_cfgs):
    if config.data.type == "mydata":
        data, modified_config = MyData(config, client_cfgs)
        return data, modified_config


register_data("mydata", call_my_data)
