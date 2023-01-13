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


class MetricCalculator(object):
    """
    Initializes the metric functions for the monitor. Use ``eval(ctx)`` \
    to get evaluation results.

    Args:
        eval_metric: set of metric names
    """
    def __init__(self, eval_metric: Union[Set[str], List[str], str]):
        # Add personalized metrics
        if isinstance(eval_metric, str):
            eval_metric = {eval_metric}
        elif isinstance(eval_metric, list):
            eval_metric = set(eval_metric)

        # Default metric is {'loss', 'avg_loss', 'total'}
        self.eval_metric = self.get_metric_funcs(eval_metric)

    def get_metric_funcs(self, eval_metric):
        """
        Build metrics for evaluation.
        Args:
            self: write your description
            eval_metric: write your description

        Returns:
        A metric calculator dict, such as \
        ``{'loss': (eval_loss, False), 'acc': (eval_acc, True), ...}``

        Note:
          The key-value pairs of built-in metric and related funcs and \
          ``the_larger_the_better`` sign is shown below:
            =================  =============================================  =
            Metric name        Source                                         \
            The larger the better
            =================  =============================================  =
            ``loss``           ``monitors.metric_calculator.eval_loss``       \
            False
            ``avg_loss``       ``monitors.metric_calculator.eval_avg_loss``   \
            False
            ``total``          ``monitors.metric_calculator.eval_total``      \
            False
            ``correct``        ``monitors.metric_calculator.eval_correct``    \
            True
            ``acc``            ``monitors.metric_calculator.eval_acc``        \
            True
            ``ap``             ``monitors.metric_calculator.eval_ap``         \
            True
            ``f1``             ``monitors.metric_calculator.eval_f1_score``   \
            True
            ``roc_auc``        ``monitors.metric_calculator.eval_roc_auc``    \
            True
            ``rmse``           ``monitors.metric_calculator.eval_rmse``       \
            False
            ``mse``            ``monitors.metric_calculator.eval_mse``        \
            False
            ``loss_regular``   ``monitors.metric_calculator.eval_regular``    \
            False
            ``imp_ratio``      ``monitors.metric_calculator.eval_imp_ratio``  \
            True
            ``std``            ``None``                                       \
            False
            ``hits@{n}``       ``monitors.metric_calculator.eval_hits``       \
            True
            =================  =============================================  =
        """
        metric_buildin = {
            metric: SUPPORT_METRICS[metric]
            for metric in {'loss', 'avg_loss', 'total'} | eval_metric
            if metric in SUPPORT_METRICS
        }
        metric_register = get_metric(eval_metric - set(SUPPORT_METRICS.keys()))
        return {**metric_buildin, **metric_register}

    def eval(self, ctx):
        results = {}
        y_true, y_pred, y_prob = self._check_and_parse(ctx)
        for metric, (func, _) in self.eval_metric.items():
            results["{}_{}".format(ctx.cur_split,
                                   metric)] = func(ctx=ctx,
                                                   y_true=y_true,
                                                   y_pred=y_pred,
                                                   y_prob=y_prob,
                                                   metric=metric)
        return results

    def _check_and_parse(self, ctx):
        """
        Check the format of the prediction and labels

        Args:
            ctx: context of trainer, see ``core.trainers.context``

        Returns:
            y_true: The ground truth labels
            y_pred: The prediction categories for classification task
            y_prob: The output of the model

        """
        if ctx.get('ys_true', None) is None:
            raise KeyError('Missing key ys_true!')
        if ctx.get('ys_prob', None) is None and \
                ctx.get('ys_pred', None) is None:
            raise KeyError('Missing key ys_prob and ys_pred!')

        y_true = ctx.ys_true
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        y_prob = None
        if ctx.get('ys_pred', None) is not None:
            y_pred = ctx.ys_pred
            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            if y_true.ndim == 1:
                y_true = np.expand_dims(y_true, axis=-1)
            if y_pred.ndim == 1:
                y_pred = np.expand_dims(y_pred, axis=-1)
        else:
            y_prob = ctx.ys_prob
            if torch is not None and isinstance(y_prob, torch.Tensor):
                y_prob = y_prob.detach().cpu().numpy()

            if 'regression' in ctx.cfg.model.task.lower():
                y_pred = None
            else:
                # classification task
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, axis=-1)
                if y_prob.ndim == 2:
                    y_prob = np.expand_dims(y_prob, axis=-1)

                # if len(y_prob.shape) > len(y_true.shape):
                y_pred = np.argmax(y_prob, axis=1)

                # check shape and type
                if not isinstance(y_true, np.ndarray):
                    raise RuntimeError('Type not support!')
                if not y_true.shape == y_pred.shape:
                    raise RuntimeError('Shape not match!')
                if not y_true.ndim == 2:
                    raise RuntimeError(
                        'y_true must be 2-dim array, {}-dim given'.format(
                            y_true.ndim))

        return y_true, y_pred, y_prob


# Metric for performance
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
            # TODO: handle missing label classes
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i],
                              softmax(y_prob[is_labeled, :, i], axis=-1),
                              multi_class='ovr'))
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


# Metric for model dissimilarity
def calc_blocal_dissim(last_model, local_updated_models):
    """
    Arguments:
        last_model (dict): the state of last round.
        local_updated_models (list): each element is (data_size, model).

    Returns:
        dict: b_local_dissimilarity, the measurements proposed in \
        "Tian Li, Anit Kumar Sahu, Manzil Zaheer, and et al. Federated \
        Optimization in Heterogeneous Networks".
    """
    # for k, v in last_model.items():
    #    print(k, v)
    # for i, elem in enumerate(local_updated_models):
    #    print(i, elem)
    local_grads = []
    weights = []
    local_gnorms = []
    for tp in local_updated_models:
        weights.append(tp[0])
        grads = dict()
        gnorms = dict()
        for k, v in tp[1].items():
            grad = v - last_model[k]
            grads[k] = grad
            gnorms[k] = torch.sum(grad**2)
        local_grads.append(grads)
        local_gnorms.append(gnorms)
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    avg_gnorms = dict()
    global_grads = dict()

    for i in range(len(local_updated_models)):
        gnorms = local_gnorms[i]
        for k, v in gnorms.items():
            if k not in avg_gnorms:
                avg_gnorms[k] = .0
            avg_gnorms[k] += weights[i] * v
        grads = local_grads[i]
        for k, v in grads.items():
            if k not in global_grads:
                global_grads[k] = torch.zeros_like(v, dtype=torch.float32)
            global_grads[k] += weights[i] * v
    b_local_dissimilarity = dict()
    for k in avg_gnorms:
        b_local_dissimilarity[k] = np.sqrt(
            avg_gnorms[k].item() / torch.sum(global_grads[k]**2).item())
    return b_local_dissimilarity


def calc_l2_dissim(last_model, local_updated_models):
    l2_dissimilarity = dict()
    l2_dissimilarity['raw'] = []
    for tp in local_updated_models:
        grads = dict()
        for key, w in tp[1].items():
            grad = w - last_model[key]
            grads[key] = grad
        grad_norm = \
            torch.norm(torch.cat([v.flatten() for v in grads.values()])).item()
        l2_dissimilarity['raw'].append(grad_norm)
    l2_dissimilarity['mean'] = np.mean(l2_dissimilarity['raw'])
    return l2_dissimilarity
