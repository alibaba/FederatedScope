import re
import string
import numpy as np
from collections import defaultdict, Counter
from federatedscope.register import register_metric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def acc_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def load_record_metrics(ctx, y_true, y_pred, y_prob, **kwargs):
    examples = ctx.get(f'{ctx.cur_split}_loader').loader.dataset
    # train mode
    if len(examples) != len(y_prob):
        return acc_score(y_true, y_pred)

    # val and test mode
    qid2pred = defaultdict(list)
    qid2ans = {}
    for prob, example in zip(y_prob, examples):
        qid = example['question_id']
        qid2pred[qid].append((prob[1], example['entity']))
        if qid not in qid2ans:
            qid2ans[qid] = example['answers']

    n_total, f1, em = 0, 0, 0
    for qid in qid2pred:
        entity = sorted(qid2pred[qid], reverse=True)[0][1]
        n_total += 1
        f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
        em += metric_max_over_ground_truths(exact_match_score, entity,
                                            qid2ans[qid])
    f1 /= n_total
    em /= n_total

    return f1


def call_record_metric(types):
    if 'record' in types:
        the_larger_the_better = True
        return 'record', load_record_metrics, the_larger_the_better


register_metric('record', call_record_metric)
