import logging
from typing import Optional, Union, List, Set

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from federatedscope.core.auxiliaries.metric_builder import get_metric

# Blind torch
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


# TODO: make this as a sub-module of monitor class
class MetricCalculator(object):
    def __init__(self, eval_metric: Union[Set[str], List[str], str]):

        # Add personalized metrics
        if isinstance(eval_metric, str):
            eval_metric = {eval_metric}
        elif isinstance(eval_metric, list):
            eval_metric = set(eval_metric)

        # Default metric is {'loss', 'avg_loss', 'total'}
        self.eval_metric = self.get_metric_funcs(eval_metric)

    def get_metric_funcs(self, eval_metric):
        metric_buildin = {
            metric: SUPPORT_METRICS[metric]
            for metric in {'loss', 'avg_loss', 'total'} | eval_metric
            if metric in SUPPORT_METRICS
        }
        metric_register = get_metric(eval_metric - set(SUPPORT_METRICS.keys()))
        return {**metric_buildin, **metric_register}

    def eval(self, ctx):
        results = {}
        y_true, y_pred = self._check_and_parse(ctx)
        for metric, (func, _) in self.eval_metric.items():
            if ctx.cur_split == 'val' and \
                    metric not in {'loss', 'avg_loss', 'total'}:
                continue

            results["{}_{}".format(ctx.cur_split,
                                   metric)] = func(ctx=ctx,
                                                   y_true=y_true,
                                                   y_pred=y_pred,
                                                   metric=metric)
        return results

    def _check_and_parse(self, ctx):
        if ctx.get('ys_true', None) is None:
            raise KeyError('Missing key ys_true!')
        if ctx.get('ys_pred', None) is None:
            raise KeyError('Missing key ys_pred!')

        y_true = ctx.ys_true
        y_pred = ctx.ys_pred

        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=-1)
        if y_pred.ndim == 1:
            y_pred = np.expand_dims(y_pred, axis=-1)

        return y_true, y_pred


def eval_correct(y_true, y_pred, **kwargs):
    correct_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        correct_list.append(np.sum(correct))
    return sum(correct_list) / len(correct_list)


def eval_acc(y_true, y_pred, **kwargs):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def eval_ap(y_true, y_pred, **kwargs):
    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        logger.warning('No positively labeled data available. ')
        return 0.0

    return sum(ap_list) / len(ap_list)


def eval_f1_score(y_true, y_pred, **kwargs):
    return f1_score(y_true, y_pred, average='macro')


def eval_hits(y_true, y_prob, metric, **kwargs):
    n = int(metric.split('@')[1])
    hits_list = []
    for i in range(y_true.shape[1]):
        idx = np.argsort(-y_prob[:, :, i], axis=1)
        pred_rank = idx.argsort(axis=1)
        # Obtain the label rank
        arg = np.arange(0, pred_rank.shape[0])
        rank = pred_rank[arg, y_true[:, i]] + 1
        hits_num = (rank <= n).sum().item()
        hits_list.append(float(hits_num) / len(rank))

    return sum(hits_list) / len(hits_list)


def eval_roc_auc(y_true, y_prob, **kwargs):
    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            y_true_one_hot = np.eye(y_prob.shape[1])[y_true[is_labeled, i]]
            rocauc_list.append(
                roc_auc_score(y_true_one_hot,
                              softmax(y_prob[is_labeled, :, i], axis=-1)))
    if len(rocauc_list) == 0:
        logger.warning('No positively labeled data available.')
        return 0.5

    return sum(rocauc_list) / len(rocauc_list)


def eval_rmse(y_true, y_prob, **kwargs):
    return np.sqrt(np.mean(np.power(y_true - y_prob, 2)))


def eval_mse(y_true, y_prob, **kwargs):
    return np.mean(np.power(y_true - y_prob, 2))


def eval_loss(ctx, **kwargs):
    return ctx.loss_batch_total


def eval_avg_loss(ctx, **kwargs):
    return ctx.loss_batch_total / ctx.num_samples


def eval_total(ctx, **kwargs):
    return ctx.num_samples


def eval_regular(ctx, **kwargs):
    return ctx.loss_regular_total


def eval_imp_ratio(ctx, y_true, y_prob, y_pred, **kwargs):
    if not hasattr(ctx.cfg.eval, 'base') or ctx.cfg.eval.base <= 0:
        logger.info(
            "To use the metric `imp_rato`, please set `eval.base` as the "
            "basic performance and it must be greater than zero.")
        return 0.

    base = ctx.cfg.eval.base
    task = ctx.cfg.model.task.lower()
    if 'regression' in task:
        perform = eval_mse(y_true, y_prob)
    elif 'classification' in task:
        perform = 1 - eval_acc(y_true, y_pred)
    return (base - perform) / base * 100.


# SUPPORT_METRICS dict, key: `metric_name`, value: (eval_func,
# the_larger_the_better)
SUPPORT_METRICS = {
    'loss': (eval_loss, False),
    'avg_loss': (eval_avg_loss, False),
    'total': (eval_total, False),
    'correct': (eval_correct, True),
    'acc': (eval_acc, True),
    'ap': (eval_ap, True),
    'f1': (eval_f1_score, True),
    'roc_auc': (eval_roc_auc, True),
    'rmse': (eval_rmse, False),
    'mse': (eval_mse, False),
    'loss_regular': (eval_regular, False),
    'imp_ratio': (eval_imp_ratio, True),
    'std': (None, False),
    **dict.fromkeys([f'hits@{n}' for n in range(1, 101)], (eval_hits, True))
}
