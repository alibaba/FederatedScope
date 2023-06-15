import re
import logging
import numpy as np
from itertools import chain
from datasets import load_dataset
from federatedscope.nlp.prompt_tuning.dataset.utils import \
    NUM_DEBUG, map_dataset_name_and_config

logger = logging.getLogger(__name__)


class PLDataProcessor(object):
    def __init__(self, config, tokenizer, debug=False):
        self.tokenizer = tokenizer
        self.debug = debug
        self.dataset_name = config.data.dataset_name
        self.make_global_train = config.federate.make_global_train
        if self.make_global_train:  # server & client
            self.num_clients = config.federate.client_num + 1
        else:
            self.num_clients = config.federate.client_num
        self.train_frac = config.data.train_frac
        self.block_size = config.data.max_seq_len

        dataset_name, dataset_config_name = map_dataset_name_and_config(
            self.dataset_name)
        raw_dataset = load_dataset(dataset_name, dataset_config_name)
        if 'validation' not in raw_dataset.keys():
            raw_dataset['train'] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f'train[:{int(self.train_frac * 100)}%]',
            )
            raw_dataset['validation'] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f'train[{int(self.train_frac * 100)}%:]',
            )
        tok_dataset = raw_dataset.map(
            self.tokenize_func,
            batched=True,
            load_from_cache_file=True,
            remove_columns=raw_dataset['train'].column_names,
        )
        max_length = max([
            max([len(x['input_ids']) for x in v])
            for v in tok_dataset.values()
        ])
        max_length = (max_length // 8 + 1) * 8  # pad to the multiple of 8
        self.max_length = min(self.block_size, max_length)
        self.dataset = tok_dataset.map(
            self.pad_func,
            batched=True,
            load_from_cache_file=True,
        )

    def tokenize_func(self, examples):
        if self.dataset_name == 'wikitext':
            return self.tokenizer(examples['text'])
        else:
            if self.dataset_name in {'arc_challenge', 'arc_easy'}:
                _template = 'Question: {}\nAnswer:'
                ctx = examples['question']
                context = [_template.format(c) for c in ctx]

                choices = examples['choices']
                answers = examples['answerKey']
                num_to_letter = {
                    '1': 'A',
                    '2': 'B',
                    '3': 'C',
                    '4': 'D',
                    '5': 'E'
                }
                for idx, answer in enumerate(answers):
                    answer = num_to_letter.get(answer, answer)
                    answer = ord(answer) - ord('A')
                    answers[idx] = choices[idx]['text'][answer]
                target = answers

            elif self.dataset_name == 'hellaswag':

                def _preprocess(text):
                    text = text.strip()
                    # NOTE: Brackets are artifacts of the WikiHow dataset
                    # portion of HellaSwag.
                    text = text.replace(' [title]', '. ')
                    text = re.sub('\\[.*?\\]', '', text)
                    text = text.replace('  ', ' ')
                    return text

                ctx_zip = zip(examples['activity_label'], examples['ctx_a'],
                              examples['ctx_b'])
                context = [
                    _preprocess(a + ': ' + b + ' ' + c.capitalize())
                    for a, b, c in ctx_zip
                ]

                labels = examples['label']
                endings = examples['endings']
                targets = []
                for idx, label in enumerate(labels):
                    target = '' if label == '' else endings[idx][int(label)]
                    targets.append(_preprocess(target))
                target = targets

            elif self.dataset_name == 'openbookqa':
                context = examples['question_stem']
                choices = examples['choices']
                answers = examples['answerKey']
                targets = []
                for choice, answer in zip(choices, answers):
                    answer = ord(answer.strip()) - ord('A')
                    targets.append(choice['text'][answer])
                target = targets

            elif self.dataset_name == 'piqa':
                _template = 'Question: {}\nAnswer:'
                ctx = examples['goal']
                context = [_template.format(c) for c in ctx]

                if -1 in examples['label']:  # test set
                    target = [''] * len(examples['label'])
                else:
                    gt_tuples = [('sol{}'.format(label + 1), idx)
                                 for idx, label in enumerate(examples['label'])
                                 ]
                    target = [examples[k][i] for k, i in gt_tuples]

            elif self.dataset_name == 'race':

                def _doc_to_text(article, question):
                    text = 'Article: ' + article + '\n\n'
                    text += 'Question: ' + question + '\n\n'
                    text += 'Answer:'
                    return text

                context = [
                    _doc_to_text(article,
                                 question) for article, question in zip(
                                     examples['article'], examples['question'])
                ]
                answers = examples['answer']
                options = examples['options']
                for idx, answer in enumerate(answers):
                    answers[idx] = options[idx][ord(answer) - ord('A')]
                target = answers

            elif self.dataset_name == 'sciq':
                _template = '{}\nQuestion: {}\nAnswer:'
                sources = examples['support']
                queries = examples['question']
                context = [
                    _template.format(s, q) for s, q in zip(sources, queries)
                ]
                target = examples['correct_answer']

            elif self.dataset_name == 'web_questions':
                context = [
                    'Question: ' + question + '\nAnswer:'
                    for question in examples['question']
                ]
                target = [' ' + answers[0] for answers in examples['answers']]

            else:
                raise KeyError(f'Dataset `{self.dataset_name}` is not '
                               f'supported.')

            context = self.tokenizer(context)
            target = self.tokenizer(target)
            # if context is ending with special token, remove it
            if len(context['input_ids'][0]) > 0 and context['input_ids'][0][
                    -1] in self.tokenizer.all_special_ids:
                context['input_ids'] = [i[:-1] for i in context['input_ids']]
                context['attention_mask'] = [
                    a[:-1] for a in context['attention_mask']
                ]
            # if target is starting with special token, remove it
            if len(target['input_ids'][0]) > 0 and target['input_ids'][0][
                    0] in self.tokenizer.all_special_ids:
                target['input_ids'] = [i[1:] for i in target['input_ids']]
                target['attention_mask'] = [
                    a[1:] for a in target['attention_mask']
                ]

            out = dict()
            out['input_ids'] = [
                i1 + i2
                for i1, i2 in zip(context['input_ids'], target['input_ids'])
            ]
            out['attention_mask'] = [
                a1 + a2 for a1, a2 in zip(context['attention_mask'],
                                          target['attention_mask'])
            ]
            # set -100 for context tokens
            out['labels'] = [
                [-100] * len(i1) + i2
                for i1, i2 in zip(context['input_ids'], target['input_ids'])
            ]

            return out

    def pad_func(self, examples):
        if self.dataset_name == 'wikitext':
            # concatenate all texts
            concatenated_examples = {
                k: list(chain(*examples[k]))
                for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= self.block_size:
                total_length = (total_length // self.block_size) * \
                               self.block_size
            # split by chunks of self.block_size
            examples = {
                k: [
                    t[i:i + self.block_size]
                    for i in range(0, total_length, self.block_size)
                ]
                for k, t in concatenated_examples.items()
            }
            examples['labels'] = examples['input_ids'].copy()

        else:
            # pad sequences
            examples['input_ids'] = [
                i + [self.tokenizer.pad_token_id] * (self.max_length - len(i))
                for i in examples['input_ids']
            ]
            examples['attention_mask'] = [[1] * len(i) + [0] *
                                          (self.max_length - len(i))
                                          for i in examples['attention_mask']]
            examples['labels'] = [
                i + [-100] * (self.max_length - len(i))
                for i in examples['labels']
            ]
            # truncate sequences
            examples['input_ids'] = [
                i[:self.max_length] for i in examples['input_ids']
            ]
            examples['attention_mask'] = [
                a[:self.max_length] for a in examples['attention_mask']
            ]
            examples['labels'] = [
                label[:self.max_length] for label in examples['labels']
            ]

        return examples

    def split_data(self):
        train_data = self.dataset['train'][:]
        val_data = self.dataset['validation'][:]
        test_data = self.dataset['test'][:]

        indices = np.arange(0, len(train_data['input_ids']))
        np.random.shuffle(indices)
        idx_slice = [
            x.tolist() for x in np.array_split(indices, self.num_clients)
        ]
        train_data = [{
            k: np.array(v)[idxs].tolist()
            for k, v in train_data.items()
        } for idxs in idx_slice]

        if self.debug:
            train_data = [{k: v[:NUM_DEBUG]
                           for k, v in data.items()} for data in train_data]
            val_data = {k: v[:NUM_DEBUG] for k, v in val_data.items()}
            test_data = {k: v[:NUM_DEBUG] for k, v in test_data.items()}

        if not self.make_global_train:
            train_data = [None] + train_data

        return train_data, val_data, test_data
