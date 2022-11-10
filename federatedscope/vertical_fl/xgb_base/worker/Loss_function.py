import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class TwoClassification:
    """
    y = {1, 0}
    L = -yln(p)-(1-y)ln(1-p)
    """
    def metric(self, y, y_pred):
        yy = 1.0 / (1.0 + np.exp(-y_pred))
        yy[yy >= 0.5] = 1.
        yy[yy < 0.5] = 0
        acc = np.sum(yy == y) / len(y)
        return 'acc', acc

    def loss(self, y, y_pred):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        res = np.mean(-y * np.log(y_pred))
        return res

    def get_grad_and_hess(self, y, pred):
        pred = np.asarray(pred)
        y = np.array(y)
        prob = 1.0 / (1.0 + np.exp(-pred))
        grad = prob - y
        hess = prob * (1.0 - prob)
        return grad, hess


class Regression_by_mae:
    def metric(self, y, y_pred):
        return 'mae', np.mean(np.abs(y - y_pred))

    def loss(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))

    def get_grad_and_hess(y, y_pred):
        x = y_pred - y
        grad = np.sign(x)
        hess = np.zeros_like(x)
        return grad, hess


class Regression_by_mse:
    def metric(self, y, y_pred):
        return 'mse', np.mean((y - y_pred)**2)

    def loss(self, y, y_pred):
        return np.mean((y - y_pred)**2)

    def get_grad_and_hess(self, y, y_pred):
        x = y_pred - y
        grad = x
        hess = np.ones_like(x)
        return grad, hess
