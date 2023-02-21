from federatedscope.vertical_fl.trainer import VerticalTrainer, \
    RandomForestTrainer


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
        from federatedscope.vertical_fl.trainer import \
            createFeatureOrderProtectedTrainer
        return createFeatureOrderProtectedTrainer(cls=trainer_cls,
                                                  model=model,
                                                  data=data,
                                                  device=device,
                                                  config=config,
                                                  monitor=monitor)
    elif protect_object in ['grad_and_hess']:
        from federatedscope.vertical_fl.trainer import \
            createLabelProtectedTrainer
        return createLabelProtectedTrainer(cls=trainer_cls,
                                           model=model,
                                           data=data,
                                           device=device,
                                           config=config,
                                           monitor=monitor)
    else:
        raise ValueError


def bucketize(feature_order, bucket_size, bucket_num):
    if isinstance(bucket_size, int):
        bucket_size = [bucket_size for _ in range(bucket_num)]

    bucketized_feature_order = list()
    start = 0
    for each_bucket_size in bucket_size:
        end = start + each_bucket_size
        bucketized_feature_order.append(feature_order[start:end])
        start = end
    return bucketized_feature_order
