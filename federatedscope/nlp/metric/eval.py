"""
The implementations are adapted from https://github.com/hugochan/
RL-based-Graph2Seq-for-NQG/blob/master/src/core/evaluation/eval.py and
https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
"""

from json import encoder
from collections import defaultdict
from federatedscope.nlp.metric.bleu import Bleu
from federatedscope.nlp.metric.meteor import Meteor

encoder.FLOAT_REPR = lambda o: format(o, '.4f')


class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self, include_meteor=True, verbose=False):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        ]
        if include_meteor:
            scorers.append((Meteor(), "METEOR"))

        for scorer, method in scorers:
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if verbose:
                        print("%s: %0.5f" % (m, sc))
                    output[m] = sc
            else:
                if verbose:
                    print("%s: %0.5f" % (method, score))
                output[method] = score
        return output


def eval(out_file, src_file, tgt_file):
    pairs = []
    with open(src_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, 'r', encoding='utf-8') as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)

    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        res[key] = [pair['prediction']]
        gts[key].append(pair['tokenized_question'])

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-out",
                        "--out_file",
                        dest="out_file",
                        default="./output/pred.txt",
                        help="output file to compare")
    parser.add_argument("-src",
                        "--src_file",
                        dest="src_file",
                        default="../data/processed/src-test.txt",
                        help="src file")
    parser.add_argument("-tgt",
                        "--tgt_file",
                        dest="tgt_file",
                        default="../data/processed/tgt-test.txt",
                        help="target file")
    args = parser.parse_args()

    print("scores: \n")
    eval(args.out_file, args.src_file, args.tgt_file)
