import numpy as np


class BinaryClsLoss(object):
    """
    y = {1, 0}
    L = -yln(p)-(1-y)ln(1-p)
    """
    def get_metric(self, y, y_pred):
        pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
        pred_prob[pred_prob >= 0.5] = 1.
        pred_prob[pred_prob < 0.5] = 0
        acc = np.sum(pred_prob == y) / len(y)
        return {'acc': acc}

    def get_loss(self, y, y_pred):
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
