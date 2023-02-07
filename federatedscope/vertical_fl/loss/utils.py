def get_vertical_loss(loss_type):
    if loss_type == 'CrossEntropyLoss':
        from federatedscope.vertical_fl.loss import BinaryClsLoss
        return BinaryClsLoss()
    elif loss_type == 'RegressionMSELoss':
        from federatedscope.vertical_fl.loss import RegressionMSELoss
        return RegressionMSELoss()
    elif loss_type == 'RegressionMAELoss':
        from federatedscope.vertical_fl.loss import RegressionMAELoss
        return RegressionMAELoss()
