from scipy.stats import pearsonr, spearmanr


def compute_sts_metrics(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {'pearsonr': pearson_corr, 'spearmanr': spearman_corr, 'corr': (pearson_corr + spearman_corr) / 2}
