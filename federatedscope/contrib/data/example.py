from federatedscope.register import register_data


def MyData(config):
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
    return data, config


def call_my_data(config):
    if config.data.type == "mydata":
        data, modified_config = MyData(config)
        return data, modified_config


register_data("mydata", call_my_data)
