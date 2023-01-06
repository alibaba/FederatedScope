"""
The implementations are adapted from https://github.com/tylin/coco-caption/
blob/master/pycocoevalcap/bleu/bleu.py
"""

from federatedscope.nlp.metric.bleu.bleu_scorer import BleuScorer


class Bleu(object):
    """
    The implementation of BLEU refer to 'Bleu: a method for automatic
    evaluation of machine translation.' [Papineni et al., 2002]
    (https://aclanthology.org/P02-1040.pdf)
    """
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        return score, scores

    def method(self):
        return "Bleu"
