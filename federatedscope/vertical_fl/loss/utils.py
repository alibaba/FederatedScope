def get_vertical_loss(loss_type, model_type):
    if loss_type == 'CrossEntropyLoss':
        from federatedscope.vertical_fl.loss import BinaryClsLoss
        return BinaryClsLoss(model_type=model_type)
    elif loss_type == 'RegressionMSELoss':
        from federatedscope.vertical_fl.loss import RegressionMSELoss
        return RegressionMSELoss(model_type=model_type)
    elif loss_type == 'RegressionMAELoss':
        from federatedscope.vertical_fl.loss import RegressionMAELoss
        return RegressionMAELoss(model_type=model_type)
