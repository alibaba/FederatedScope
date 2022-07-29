def get_mfnet(model_config, data_shape):
    """Return the MF model according to model configs

    Arguments:
        model_config: the model related parameters
        data_shape (int): the input shape of the model
    """
    if model_config.type.lower() == 'vmfnet':
        from federatedscope.mf.model.model import VMFNet
        return VMFNet(num_user=model_config.num_user,
                      num_item=data_shape,
                      num_hidden=model_config.hidden)
    else:
        from federatedscope.mf.model.model import HMFNet
        return HMFNet(num_user=data_shape,
                      num_item=model_config.num_item,
                      num_hidden=model_config.hidden)
