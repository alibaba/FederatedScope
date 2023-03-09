import numpy as np


class BinaryClsLoss(object):
    """
    y = {1, 0}
    L = -yln(p)-(1-y)ln(1-p)
    """
    def __init__(self, model_type):
        self.cal_hess = model_type in ['xgb_tree']
        self.cal_sigmoid = model_type in ['xgb_tree', 'gbdt_tree']
        self.merged_mode = 'mean' if model_type in ['random_forest'] else 'sum'

    def _sigmoid(self, y_pred):
        return 1.0 / (1.0 + np.exp(-y_pred))

    def _process_y_pred(self, y_pred):
        if self.merged_mode == 'mean':
            y_pred = np.mean(y_pred, axis=0)
        else:
            y_pred = np.sum(y_pred, axis=0)

        if self.cal_sigmoid:
            y_pred = self._sigmoid(y_pred)

        return y_pred

    def get_metric(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        y_pred = (y_pred >= 0.5).astype(np.float32)
        acc = np.sum(y_pred == y) / len(y)
        return {'acc': acc}

    def get_loss(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        res = np.mean(-y * np.log(y_pred + 1e-7))
        return res

    def get_grad_and_hess(self, y, y_pred):
        y_pred = self._process_y_pred(y_pred)
        y = np.array(y)
        grad = y_pred - y
        hess = y_pred * (1.0 - y_pred) if self.cal_hess else None
        return grad, hess
