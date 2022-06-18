from scipy.stats import pearsonr, spearmanr
from federatedscope.register import register_metric


def compute_sts_metrics(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {'pearsonr': pearson_corr, 'spearmanr': spearman_corr, 'corr': (pearson_corr + spearman_corr) / 2}


def load_sts_metrics(y_true, y_prob, **kwargs):
    results = compute_sts_metrics(y_prob.squeeze(), y_true.squeeze())
    return results


def call_sts_metric(types):
    if 'sts' in types:
        return 'sts', load_sts_metrics


register_metric('sts', call_sts_metric)
