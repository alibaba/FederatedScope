import logging
import os
from federatedscope.register import register_metric
from federatedscope.nlp.metric.rouge.utils import test_rouge
from federatedscope.nlp.metric.eval import eval

logger = logging.getLogger(__name__)


def load_msqg_metrics(ctx, **kwargs):
    tmp_dir = os.path.join(ctx.cfg.outdir, 'temp')
    rouges = test_rouge(tmp_dir, ctx.pred_path, ctx.tgt_path)
    qg_res = eval(ctx.pred_path, ctx.src_path, ctx.tgt_path)  # bleu & meteor

    results = rouges
    results.update(qg_res)
    results = {
        k: v
        for k, v in results.items()
        if k in {'rouge_l_f_score', 'Bleu_4', 'METEOR'}
    }
    return results


def call_msqg_metric(types):
    if 'msqg' in types:
        the_larger_the_better = True
        return 'msqg', load_msqg_metrics, the_larger_the_better


register_metric('msqg', call_msqg_metric)
