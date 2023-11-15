import numpy as np


class RegressionMAELoss(object):
    def __init__(self, model_type):
        self.cal_hess = model_type in ['xgb_tree']
        self.merged_mode = 'mean' if model_type in ['random_forest'] else 'sum'

    def _process_y_pred(self, y_pred):
        if self.merged_mode == 'mean':
            y_pred = np.mean(y_pred, axis=0)
        else:
            y_pred = np.sum(y_pred, axis=0)

        return y_pred

    def get_metric(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        return {'mae': np.mean(np.abs(y - y_pred))}

    def get_loss(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        return np.mean(np.abs(y - y_pred))

    def get_grad_and_hess(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        x = y_pred - y
        grad = np.sign(x)
        hess = np.zeros_like(x) if self.cal_hess else None
        return grad, hess


class RegressionMSELoss(object):
    def __init__(self, model_type):
        self.cal_hess = model_type in ['xgb_tree']
        self.merged_mode = 'mean' if model_type in ['random_forest'] else 'sum'

    def _process_y_pred(self, y_pred):
        if self.merged_mode == 'mean':
            y_pred = np.mean(y_pred, axis=0)
        else:
            y_pred = np.sum(y_pred, axis=0)

        return y_pred

    def get_metric(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        return {'mse': np.mean((y - y_pred)**2)}

    def get_loss(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        return np.mean((y - y_pred)**2)

    def get_grad_and_hess(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        x = y_pred - y
        grad = x
        hess = np.ones_like(x) if self.cal_hess else None
        return grad, hess
