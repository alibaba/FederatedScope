from federatedscope.register import register_metric
from federatedscope.nlp.hetero_tasks.metric.squad import compute_squad_metrics


def load_newsqa_metrics(ctx, **kwargs):
    examples = ctx.get('{}_examples'.format(ctx.cur_split))
    encoded_inputs = ctx.get('{}_encoded'.format(ctx.cur_split))
    results = ctx.newsqa_results
    n_best_size = ctx.cfg.model.n_best_size
    max_answer_len = ctx.cfg.model.max_answer_len
    null_score_diff_threshold = ctx.cfg.model.null_score_diff_threshold

    metrics = compute_squad_metrics(examples, encoded_inputs, results,
                                    n_best_size, max_answer_len,
                                    null_score_diff_threshold)
    return metrics


def call_newsqa_metric(types):
    if 'newsqa' in types:
        the_larger_the_better = True
        return 'newsqa', load_newsqa_metrics, the_larger_the_better


register_metric('newsqa', call_newsqa_metric)
