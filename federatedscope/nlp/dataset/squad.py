import os
import os.path as osp
import numpy as np
import torch
import json
import logging
import urllib.request
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset
from federatedscope.nlp.auxiliaries.utils import reporthook

logger = logging.getLogger(__name__)


class SquadExample(object):
    def __init__(self, qa_id, question, context, train_answer, val_answer, start_pos, end_pos, context_tokens, is_impossible):
        self.qa_id = qa_id
        self.question = question
        self.context = context
        self.train_answer = train_answer
        self.val_answer = val_answer
        self.start_position = start_pos
        self.end_position = end_pos
        self.context_tokens = context_tokens
        self.is_impossible = is_impossible


class SquadEncodedInput(object):
    def __init__(self, token_ids, token_type_ids, attention_mask, overflow_token_ids):
        self.token_ids = token_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.overflow_token_ids = overflow_token_ids


class SquadResult(object):
    def __init__(self, unique_id, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits


def refine_subtoken_position(context_subtokens, subtoken_start_pos, subtoken_end_pos, tokenizer, annotated_answer):
    subtoken_answer = ' '.join(tokenizer.tokenize(annotated_answer))
    for new_st in range(subtoken_start_pos, subtoken_end_pos + 1):
        for new_ed in range(subtoken_end_pos, subtoken_start_pos - 1, -1):
            text_span = ' '.join(context_subtokens[new_st:(new_ed + 1)])
            if text_span == subtoken_answer:
                return new_st, new_ed
    return subtoken_start_pos, subtoken_end_pos


def get_char_to_word_positions(context, answer, start_char_pos, is_impossible):
    context_tokens = []
    char_to_word_offset = []
    is_prev_whitespace = True
    for c in context:
        is_whitespace = (c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F)
        if is_whitespace:
            is_prev_whitespace = True
        else:
            if is_prev_whitespace:
                context_tokens.append(c)
            else:
                context_tokens[-1] += c
            is_prev_whitespace = False
        char_to_word_offset.append(len(context_tokens) - 1)

    start_pos, end_pos = 0, 0
    if start_char_pos is not None and not is_impossible:
        start_pos = char_to_word_offset[start_char_pos]
        end_pos = char_to_word_offset[start_char_pos + len(answer) - 1]
    return start_pos, end_pos, context_tokens


def check_max_context_token(all_spans, cur_span_idx, pos):
    best_score, best_span_idx = None, None
    for span_idx, span in enumerate(all_spans):
        end = span.context_start_position + span.context_len - 1
        if pos < span.context_start_position or pos > end:
            continue
        num_left_context = pos - span.context_start_position
        num_right_context = end - pos
        score = min(num_left_context, num_right_context) + 0.01 * span.context_len
        if best_score is None or score > best_score:
            best_score = score
            best_span_idx = span_idx
    return cur_span_idx == best_span_idx


def encode(tokenizer, text_a, text_b, max_seq_len, max_query_len, added_trunc_size):
    def _get_token_ids(text):
        if isinstance(text, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
            return tokenizer.convert_tokens_to_ids(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
            return text
        else:
            raise ValueError('Input is not valid, should be a string, a list/tuple of strings or a list/tuple of integers.')

    token_ids_a = _get_token_ids(text_a)
    token_ids_b = _get_token_ids(text_b)

    # Truncate
    overflow_token_ids = None
    len_a = len(token_ids_a) + 2
    total_len = len(token_ids_a) + len(token_ids_b) + 3
    if len_a > max_query_len:
        num_remove = len_a - max_query_len
        token_ids_a = token_ids_a[:-num_remove]
    if total_len > max_seq_len:
        num_remove = total_len - max_seq_len
        trunc_size = min(len(token_ids_b), added_trunc_size + num_remove)
        overflow_token_ids = token_ids_b[-trunc_size:]
        token_ids_b = token_ids_b[:-num_remove]

    # Combine and pad
    token_ids = [tokenizer.cls_token_id] + token_ids_a + [tokenizer.sep_token_id]
    token_type_ids = [0] * len(token_ids)
    token_ids += token_ids_b + [tokenizer.sep_token_id]
    token_type_ids += [1] * (len(token_ids_b) + 1)
    attention_mask = [1] * len(token_ids)
    if len(token_ids) < max_seq_len:
        dif = max_seq_len - len(token_ids)
        token_ids += [tokenizer.pad_token_id] * dif
        token_type_ids += [0] * dif
        attention_mask += [0] * dif

    return SquadEncodedInput(token_ids, token_type_ids, attention_mask, overflow_token_ids)


def create_squad_examples(root, split, debug=False):
    if not osp.exists(osp.join(root, split + '.json')):
        logger.info('Downloading squad dataset')
        url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/{}-v2.0.json'.format(split)
        data_dir = root
        os.makedirs(data_dir, exist_ok=True)
        data_file = osp.join(data_dir, '{}.json'.format(split))
        urllib.request.urlretrieve(url, data_file, reporthook)

    with open(osp.join(root, split + '.json'), 'r', encoding='utf-8') as reader:
        data = json.load(reader)['data']

    examples = []
    for i, entry in tqdm(enumerate(data), total=len(data)):
        if debug and len(examples) > 20:
            break
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qa_id = qa['id']
                question = qa['question']
                start_char_pos = None
                train_answer = None
                val_answer = []

                if 'is_impossible' in qa:
                    is_impossible = qa['is_impossible']
                else:
                    is_impossible = False
                if not is_impossible:
                    if split == 'train':
                        train_answer = qa['answers'][0]['text']
                        start_char_pos = qa['answers'][0]['answer_start']
                    else:
                        val_answer = qa['answers']

                start_pos, end_pos, context_tokens = get_char_to_word_positions(
                    context, train_answer, start_char_pos, is_impossible)
                examples.append(SquadExample(qa_id, question, context, train_answer, val_answer,
                                             start_pos, end_pos, context_tokens, is_impossible))

    if debug:
        examples = examples[:20]

    if split == 'train':
        num_train_samples = int(0.9 * len(examples))
        return examples[:num_train_samples], examples[num_train_samples:]
    elif split == 'dev':
        return examples


def create_squad_dataset(root, split, tokenizer, max_seq_len, max_query_len, trunc_stride, model_type,
                         cache_dir='', debug=False):
    logger.info('Preprocessing {} {} dataset'.format('squad', split))
    cache_file = osp.join(cache_dir, 'squad', '_'.join(['squad', split, str(max_seq_len), str(max_query_len),
                                                        str(trunc_stride), model_type]) + '.pt')
    if osp.exists(cache_file):
        logger.info('Loading cache file from \'{}\''.format(cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        examples = create_squad_examples(root, split, debug)
        encoded_inputs = None

    def _create_dataset(examples_, encoded_inputs_=None):
        if encoded_inputs_ is None:
            unique_id = 1000000000
            encoded_inputs_ = []
            for example_idx, example in tqdm(enumerate(examples_), total=len(examples_)):
                if split == 'train' and not example.is_impossible:
                    start_pos = example.start_position
                    end_pos = example.end_position

                    actual_answer = ' '.join(example.context_tokens[start_pos:(end_pos + 1)])
                    cleaned_answer = ' '.join(example.train_answer.strip().split())
                    if actual_answer.find(cleaned_answer) == -1:
                        logger.info('Could not find answer: {} vs. {}'.format(actual_answer, cleaned_answer))
                        continue

                tok_to_subtok_idx = []
                subtok_to_tok_idx = []
                context_subtokens = []
                for i, token in enumerate(example.context_tokens):
                    tok_to_subtok_idx.append(len(context_subtokens))
                    subtokens = tokenizer.tokenize(token)
                    for subtoken in subtokens:
                        subtok_to_tok_idx.append(i)
                        context_subtokens.append(subtoken)

                if split == 'train' and not example.is_impossible:
                    subtoken_start_pos = tok_to_subtok_idx[example.start_position]
                    if example.end_position < len(example.context_tokens) - 1:
                        subtoken_end_pos = tok_to_subtok_idx[example.end_position + 1] - 1
                    else:
                        subtoken_end_pos = len(context_subtokens) - 1
                    subtoken_start_pos, subtoken_end_pos = refine_subtoken_position(
                        context_subtokens, subtoken_start_pos, subtoken_end_pos, tokenizer, example.train_answer)

                truncated_context = context_subtokens
                len_question = min(len(tokenizer.tokenize(example.question)), max_query_len - 2)
                added_trunc_size = max_seq_len - trunc_stride - len_question - 3

                spans = []
                while len(spans) * trunc_stride < len(context_subtokens):
                    text_a = example.question
                    text_b = truncated_context
                    encoded_input = encode(tokenizer, text_a, text_b, max_seq_len, max_query_len, added_trunc_size)

                    context_start_pos = len(spans) * trunc_stride
                    context_len = min(len(context_subtokens) - context_start_pos, max_seq_len - len_question - 3)
                    context_end_pos = context_start_pos + context_len - 1

                    if tokenizer.pad_token_id in encoded_input.token_ids:
                        non_padded_ids = encoded_input.token_ids[:encoded_input.token_ids.index(tokenizer.pad_token_id)]
                    else:
                        non_padded_ids = encoded_input.token_ids
                    tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

                    context_subtok_to_tok_idx = {}
                    for i in range(context_len):
                        context_idx = len_question + i + 2
                        context_subtok_to_tok_idx[context_idx] = subtok_to_tok_idx[context_start_pos + i]

                    # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                    cls_index = encoded_input.token_ids.index(tokenizer.cls_token_id)
                    p_mask = np.array(encoded_input.token_type_ids)
                    p_mask = np.minimum(p_mask, 1)
                    p_mask[np.where(np.array(encoded_input.token_ids) == tokenizer.sep_token_id)[0]] = 1
                    p_mask[cls_index] = 0

                    start_pos, end_pos = 0, 0
                    span_is_impossible = example.is_impossible
                    if split == 'train' and not span_is_impossible:
                        # For training, if our document chunk does not contain an annotation
                        # we throw it out, since there is nothing to predict.
                        if subtoken_start_pos >= context_start_pos and subtoken_end_pos <= context_end_pos:
                            context_offset = len_question + 2
                            start_pos = subtoken_start_pos - context_start_pos + context_offset
                            end_pos = subtoken_end_pos - context_start_pos + context_offset
                        else:
                            start_pos = cls_index
                            end_pos = cls_index
                            span_is_impossible = True

                    encoded_input.start_position = start_pos
                    encoded_input.end_position = end_pos
                    encoded_input.cls_index = cls_index
                    encoded_input.p_mask = p_mask.tolist()
                    encoded_input.is_impossible = span_is_impossible

                    # For computing metrics
                    encoded_input.example_index = example_idx
                    encoded_input.context_start_position = context_start_pos
                    encoded_input.context_len = context_len
                    encoded_input.tokens = tokens
                    encoded_input.context_subtok_to_tok_idx = context_subtok_to_tok_idx
                    encoded_input.is_max_context_token = {}
                    encoded_input.unique_id = unique_id
                    spans.append(encoded_input)
                    unique_id += 1

                    if encoded_input.overflow_token_ids is None:
                        break
                    truncated_context = encoded_input.overflow_token_ids

                for span_idx in range(len(spans)):
                    for context_idx in range(spans[span_idx].context_len):
                        is_max_context_token = check_max_context_token(spans, span_idx, span_idx * trunc_stride + context_idx)
                        idx = len_question + context_idx + 2
                        spans[span_idx].is_max_context_token[idx] = is_max_context_token
                encoded_inputs_.extend(spans)

        token_ids = torch.LongTensor([inp.token_ids for inp in encoded_inputs_])
        token_type_ids = torch.LongTensor([inp.token_type_ids for inp in encoded_inputs_])
        attention_mask = torch.LongTensor([inp.attention_mask for inp in encoded_inputs_])
        start_positions = torch.LongTensor([inp.start_position for inp in encoded_inputs_])
        end_positions = torch.LongTensor([inp.end_position for inp in encoded_inputs_])

        example_indices = torch.arange(token_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(token_ids, token_type_ids, attention_mask, start_positions,
                                end_positions, example_indices)
        return dataset, encoded_inputs_, examples_

    if split == 'train':
        if encoded_inputs is not None:
            return _create_dataset(examples[0], encoded_inputs[0]), _create_dataset(examples[1], encoded_inputs[1])
        else:
            train_dataset, train_encoded, train_examples = _create_dataset(examples[0])
            val_dataset, val_encoded, val_examples = _create_dataset(examples[1])
            if cache_dir:
                logger.info('Saving cache file to \'{}\''.format(cache_file))
                os.makedirs(osp.join(cache_dir, 'squad'), exist_ok=True)
                torch.save({'examples': examples,
                            'encoded_inputs': [train_encoded, val_encoded]}, cache_file)

        return (train_dataset, train_encoded, train_examples), (val_dataset, val_encoded, val_examples)

    elif split == 'dev':
        if encoded_inputs is not None:
            return _create_dataset(examples, encoded_inputs)
        else:
            test_dataset, test_encoded, test_examples = _create_dataset(examples)
            if cache_dir:
                logger.info('Saving cache file to \'{}\''.format(cache_file))
                os.makedirs(osp.join(cache_dir, 'squad'), exist_ok=True)
                torch.save({'examples': examples,
                            'encoded_inputs': test_encoded}, cache_file)

        return test_dataset, test_encoded, test_examples
