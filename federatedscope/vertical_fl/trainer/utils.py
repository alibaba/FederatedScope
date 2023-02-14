from federatedscope.vertical_fl.trainer import VerticalTrainer, \
    FeatureOrderProtectedTrainer, RandomForestTrainer


def get_vertical_trainer(config, model, data, device, monitor):

    protect_object = config.vertical.protect_object
    if config.model.type.lower() == 'random_forest':
        return RandomForestTrainer(model=model,
                                   data=data,
                                   device=device,
                                   config=config,
                                   monitor=monitor)
    if not protect_object or protect_object == '':
        return VerticalTrainer(model=model,
                               data=data,
                               device=device,
                               config=config,
                               monitor=monitor)
    elif protect_object == 'feature_order':
        return FeatureOrderProtectedTrainer(model=model,
                                            data=data,
                                            device=device,
                                            config=config,
                                            monitor=monitor)
    else:
        raise ValueError
