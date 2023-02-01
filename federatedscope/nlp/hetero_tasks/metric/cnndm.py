import logging
import os
from federatedscope.register import register_metric
from federatedscope.nlp.metric.rouge.utils import test_rouge

logger = logging.getLogger(__name__)


def load_cnndm_metrics(ctx, **kwargs):
    tmp_dir = os.path.join(ctx.cfg.outdir, 'temp')
    rouges = test_rouge(tmp_dir, ctx.pred_path, ctx.tgt_path)
    results = {
        k: v
        for k, v in rouges.items()
        if k in {'rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score'}
    }
    return results


def call_cnndm_metric(types):
    if 'cnndm' in types:
        the_larger_the_better = True
        return 'cnndm', load_cnndm_metrics, the_larger_the_better


register_metric('cnndm', call_cnndm_metric)
