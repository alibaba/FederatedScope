def get_mfnet(model_config, local_data):
    """Return the MF model according to model configs

    Arguments:
        model_config: the model related parameters
        local_data (dict): the dataset used for this model
    """
    if model_config.type.lower() == 'vmfnet':
        from federatedscope.mf.model.model import VMFNet
        return VMFNet(num_user=model_config.num_user,
                      num_item=local_data["train"].n_col,
                      num_hidden=model_config.hidden)
    else:
        from federatedscope.mf.model.model import HMFNet
        return HMFNet(num_user=local_data["train"].n_row,
                      num_item=model_config.num_item,
                      num_hidden=model_config.hidden)
