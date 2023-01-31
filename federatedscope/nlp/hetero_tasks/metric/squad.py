import math
import collections
import string
import re
import logging
from federatedscope.register import register_metric

logger = logging.getLogger(__name__)


def normalize_answer(s):
    '''Lower text and remove punctuation, articles and extra whitespace.'''
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds):
    '''
    Computes the exact and f1 scores from the examples and the model
    predictions
    '''
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qa_id = example.qa_id
        gold_answers = [
            answer['text'] for answer in example.val_answer
            if normalize_answer(answer['text'])
        ]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = ['']

        if qa_id not in preds:
            print('Missing prediction for %s' % qa_id)
            continue

        prediction = preds[qa_id]
        exact_scores[qa_id] = max(
            compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qa_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        exact = 100.0 * sum(exact_scores.values()) / total
        f1 = 100.0 * sum(f1_scores.values()) / total
        return collections.OrderedDict([
            ('exact', exact),
            ('f1', f1),
            ('exact_and_f1', (exact + f1) / 2),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        exact = 100.0 * sum(exact_scores[k] for k in qid_list) / total
        f1 = 100.0 * sum(f1_scores[k] for k in qid_list) / total
        return collections.OrderedDict([
            ('exact', exact),
            ('f1', f1),
            ('exact_and_f1', (exact + f1) / 2),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs,
                         qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs,
                                                qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs,
                                          qid_to_has_ans)

    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def squad_evaluate(examples,
                   preds,
                   no_answer_probs=None,
                   no_answer_probability_threshold=1.0):
    qa_id_to_has_answer = {
        example.qa_id: bool(example.val_answer)
        for example in examples
    }
    has_answer_qids = [
        qa_id for qa_id, has_answer in qa_id_to_has_answer.items()
        if has_answer
    ]
    no_answer_qids = [
        qa_id for qa_id, has_answer in qa_id_to_has_answer.items()
        if not has_answer
    ]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(exact, no_answer_probs,
                                             qa_id_to_has_answer,
                                             no_answer_probability_threshold)
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs,
                                          qa_id_to_has_answer,
                                          no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold,
                                      f1_threshold,
                                      qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, 'HasAns')

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold,
                                     f1_threshold,
                                     qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, 'NoAns')

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs,
                             qa_id_to_has_answer)

    return evaluation


def get_final_text(pred_text, orig_text):
    '''Project the tokenized prediction back to the original text.'''

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding
    # to the span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra ''s'.
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is 'Steve Smith'.
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic
    # between `pred_text` and `orig_text` to get a character-to-character
    # alignment. This can fail in certain cases in which case we just return
    # `orig_text`.

    from transformers import BasicTokenizer

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer()
    tok_text = ' '.join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def get_topk_indices(logits, n_best_size):
    index_and_score = sorted(enumerate(logits),
                             key=lambda x: x[1],
                             reverse=True)

    topk_indices = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        topk_indices.append(index_and_score[i][0])
    return topk_indices


def _compute_softmax(scores):
    '''Compute softmax probability over raw logits.'''
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def create_squad_answer_texts(examples, encoded_inputs, results, n_best_size,
                              max_answer_len, null_score_diff_threshold):
    _PrelimPrediction = collections.namedtuple('PrelimPrediction', [
        'feature_index', 'start_index', 'end_index', 'start_logit', 'end_logit'
    ])
    _NbestPrediction = collections.namedtuple(
        'NbestPrediction', ['text', 'start_logit', 'end_logit'])

    example_index_to_features = collections.defaultdict(list)
    for feature in encoded_inputs:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in results:
        unique_id_to_result[result.unique_id] = result

    predicted_answer_texts = collections.OrderedDict()
    for (example_index, example) in enumerate(examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null
        # score
        null_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = get_topk_indices(result.start_logits, n_best_size)
            end_indexes = get_topk_indices(result.end_logits, n_best_size)

            # if we could have irrelevant answers, get the min score of
            # irrelevant
            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions,
                    # e.g., predict that the start of the span is in the
                    # question. We throw out all invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.context_subtok_to_tok_idx:
                        continue
                    if end_index not in feature.context_subtok_to_tok_idx:
                        continue
                    if not feature.is_max_context_token.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_len:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions.append(
            _PrelimPrediction(feature_index=min_null_feature_index,
                              start_index=0,
                              end_index=0,
                              start_logit=null_start_logit,
                              end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x:
                                    (x.start_logit + x.end_logit),
                                    reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = \
                    feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = \
                    feature.context_subtok_to_tok_idx[pred.start_index]
                orig_doc_end = \
                    feature.context_subtok_to_tok_idx[pred.end_index]
                orig_tokens = \
                    example.context_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = ' '.join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(' ##', '')
                tok_text = tok_text.replace('##', '')

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = ' '.join(tok_text.split())
                orig_text = ' '.join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ''
                seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(text=final_text,
                                 start_logit=pred.start_logit,
                                 end_logit=pred.end_logit))

        # if we didn't include the empty option in the n-best, include it
        if '' not in seen_predictions:
            nbest.append(
                _NbestPrediction(text='',
                                 start_logit=null_start_logit,
                                 end_logit=null_end_logit))

        # In very rare edge cases we could only have single null prediction.
        # So we just create a nonce prediction in this case to avoid failure.
        if len(nbest) == 1:
            nbest.insert(
                0,
                _NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        score_diff = \
            score_null - best_non_null_entry.start_logit - \
            best_non_null_entry.end_logit
        if score_diff > null_score_diff_threshold:
            predicted_answer_texts[example.qa_id] = ''
        else:
            predicted_answer_texts[example.qa_id] = best_non_null_entry.text

    return predicted_answer_texts


def compute_squad_metrics(examples,
                          encoded_inputs,
                          results,
                          n_best_size,
                          max_answer_len,
                          null_score_diff_threshold=None,
                          return_text=False):
    predicted_answer_texts = create_squad_answer_texts(
        examples, encoded_inputs, results, n_best_size, max_answer_len,
        null_score_diff_threshold)
    raw_metrics = squad_evaluate(examples, predicted_answer_texts)
    metrics = {
        k: v
        for k, v in raw_metrics.items() if k in ('exact', 'f1', 'exact_and_f1')
    }

    if return_text:
        return predicted_answer_texts
    return metrics


def load_squad_metrics(ctx, **kwargs):
    examples = ctx.get('{}_examples'.format(ctx.cur_split))
    encoded_inputs = ctx.get('{}_encoded'.format(ctx.cur_split))
    results = ctx.squad_results
    n_best_size = ctx.cfg.model.n_best_size
    max_answer_len = ctx.cfg.model.max_answer_len
    null_score_diff_threshold = ctx.cfg.model.null_score_diff_threshold

    metrics = compute_squad_metrics(examples, encoded_inputs, results,
                                    n_best_size, max_answer_len,
                                    null_score_diff_threshold)
    return metrics


def call_squad_metric(types):
    if 'squad' in types:
        the_larger_the_better = True
        return 'squad', load_squad_metrics, the_larger_the_better


register_metric('squad', call_squad_metric)
