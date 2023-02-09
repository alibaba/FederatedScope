def get_vertical_loss(loss_type, cal_hess=True):
    if loss_type == 'CrossEntropyLoss':
        from federatedscope.vertical_fl.loss import BinaryClsLoss
        return BinaryClsLoss(cal_hess=cal_hess)
    elif loss_type == 'RegressionMSELoss':
        from federatedscope.vertical_fl.loss import RegressionMSELoss
        return RegressionMSELoss(cal_hess=cal_hess)
    elif loss_type == 'RegressionMAELoss':
        from federatedscope.vertical_fl.loss import RegressionMAELoss
        return RegressionMAELoss(cal_hess=cal_hess)
