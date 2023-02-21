from federatedscope.vertical_fl.trainer import VerticalTrainer, \
    RandomForestTrainer, createFeatureOrderProtectedTrainer


def get_vertical_trainer(config, model, data, device, monitor):

    if config.model.type.lower() == 'random_forest':
        trainer_cls = RandomForestTrainer
    else:
        trainer_cls = VerticalTrainer

    protect_object = config.vertical.protect_object
    if not protect_object or protect_object == '':
        return trainer_cls(model=model,
                           data=data,
                           device=device,
                           config=config,
                           monitor=monitor)
    elif protect_object == 'feature_order':
        return createFeatureOrderProtectedTrainer(cls=trainer_cls,
                                                  model=model,
                                                  data=data,
                                                  device=device,
                                                  config=config,
                                                  monitor=monitor)
    else:
        raise ValueError
