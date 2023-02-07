import numpy as np


class RegressionMAELoss(object):
    def get_metric(self, y, y_pred):
        return {'mae': np.mean(np.abs(y - y_pred))}

    def get_loss(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))

    def get_grad_and_hess(y, y_pred):
        x = y_pred - y
        grad = np.sign(x)
        hess = np.zeros_like(x)
        return grad, hess


class RegressionMSELoss(object):
    def get_metric(self, y, y_pred):
        return {'mse': np.mean((y - y_pred)**2)}

    def get_loss(self, y, y_pred):
        return np.mean((y - y_pred)**2)

    def get_grad_and_hess(self, y, y_pred):
        x = y_pred - y
        grad = x
        hess = np.ones_like(x)
        return grad, hess
