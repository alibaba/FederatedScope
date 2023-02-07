from federatedscope.vertical_fl.model \
    import MultipleXGBTrees, MultipleGBDTTrees


def get_tree_model(model_config):

    if model_config.type.lower() == 'xgb_tree':
        return MultipleXGBTrees(max_depth=model_config.max_tree_depth,
                                lambda_=model_config.lambda_,
                                gamma=model_config.gamma,
                                num_of_trees=model_config.num_of_trees)
    elif model_config.type.lower() == 'gbdt_tree':
        return MultipleGBDTTrees(max_depth=model_config.max_tree_depth,
                                 lambda_=model_config.lambda_,
                                 gamma=model_config.gamma,
                                 num_of_trees=model_config.num_of_trees)
