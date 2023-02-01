import logging
import torch
import numpy as np
from datasets.load import load_dataset
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice
from federatedscope.nlp.prompt_learning.dataset.utils import \
    DatasetDict, NUM_DEBUG, SERVER_TRAIN

logger = logging.getLogger(__name__)


class PLDataProcessor(object):
    def __init__(self, config, train_frac=0.9):
        self.dataset_name = config.data.dataset_name
        if SERVER_TRAIN:  # server & client
            self.num_clients = config.federate.client_num + 1
        else:
            self.num_clients = config.federate.client_num
        self.train_frac = train_frac
        raw_dataset = load_dataset('super_glue', self.dataset_name)
        self.dataset = raw_dataset.map(
            self.load_data,
            batched=True,
            load_from_cache_file=False,
            remove_columns=raw_dataset['train'].column_names
            if self.dataset_name == 'record' else None,
        )

    def load_data(self, examples):
        processed_data = {'text_a': None, 'text_b': None}

        if self.dataset_name == 'boolq':
            processed_data['text_a'] = examples['question']
            processed_data['text_b'] = examples['passage']
            processed_data['labels'] = examples['label']

        elif self.dataset_name in {'cb', 'rte'}:
            processed_data['text_a'] = examples['premise']
            processed_data['text_b'] = examples['hypothesis']
            processed_data['labels'] = examples['label']

        elif self.dataset_name == 'copa':
            processed_data['text_a'] = []
            for premise, question in zip(examples['premise'],
                                         examples['question']):
                joiner = 'because' if question == 'cause' else 'so'
                processed_data['text_a'].append(f'{premise} {joiner}')

            processed_data['text_b'] = []
            for choice1, choice2 in zip(examples['choice1'],
                                        examples['choice2']):
                processed_data['text_b'].append((choice1, choice2))

            processed_data['labels'] = examples['label']

        elif self.dataset_name == 'multirc':
            processed_data['text_a'] = examples['paragraph']
            processed_data['text_b'] = []
            for question, answer in zip(examples['question'],
                                        examples['answer']):
                processed_data['text_b'].append(f'{question} {answer}')
            processed_data['labels'] = examples['label']

        elif self.dataset_name == 'record':
            processed_data['text_a'] = []
            processed_data['text_b'] = []
            processed_data['labels'] = []
            processed_data['index'] = []
            processed_data['question_id'] = []
            processed_data['entity'] = []
            processed_data['answers'] = []

            for idx, passage in enumerate(examples['passage']):
                query, entities, answers = examples['query'][idx], \
                                           examples['entities'][idx], \
                                           examples['answers'][idx]
                index = examples['idx'][idx]
                passage = passage.replace('@highlight\n', '- ')

                for ent_idx, ent in enumerate(entities):
                    question = query.replace('@placeholder', ent)
                    label = 1 if ent in answers else 0
                    processed_data['text_a'].append(passage)
                    processed_data['text_b'].append(question)
                    processed_data['labels'].append(label)
                    processed_data['index'].append(index)
                    processed_data['question_id'].append(index['query'])
                    processed_data['entity'].append(ent)
                    processed_data['answers'].append(answers)

        elif self.dataset_name == 'wic':
            processed_data['text_a'] = []
            processed_data['text_b'] = []
            for word, sent1, sent2 in zip(examples['word'],
                                          examples['sentence1'],
                                          examples['sentence2']):
                processed_data['text_a'].append(word + ': ' + sent1)
                processed_data['text_b'].append(word + ': ' + sent2)
            processed_data['labels'] = examples['label']

        elif self.dataset_name == 'wsc':
            processed_data['text_a'] = []
            for span2_text, text in zip(examples['span2_text'],
                                        examples['text']):
                processed_data['text_a'].append(span2_text + ': ' + text)
            processed_data['text_b'] = examples['span1_text']
            processed_data['labels'] = examples['label']

        else:
            raise KeyError(f'No dataset named {self.dataset_name}')

        return processed_data

    def split_data(self):
        train_val_data = self.dataset['train'][:]
        test_data = self.dataset['validation'][:]

        num_train = int(len(train_val_data['labels']) * self.train_frac)
        train_data = {k: v[:num_train] for k, v in train_val_data.items()}
        val_data = {k: v[num_train:] for k, v in train_val_data.items()}

        labels = np.array(train_data['labels'])
        idx_slice = dirichlet_distribution_noniid_slice(
            label=labels, client_num=self.num_clients, alpha=0.5)
        train_data = [{
            k: np.array(v)[idxs].tolist()
            for k, v in train_data.items()
        } for idxs in idx_slice]
        if not SERVER_TRAIN:
            train_data = [None] + train_data

        return train_data, val_data, test_data


def create_pl_dataset(data, tokenizer, dataset_name, max_seq_len, debug=False):
    if debug:
        data = {k: v[:NUM_DEBUG] for k, v in data.items()}

    text_a, text_b, labels = data['text_a'], data['text_b'], data['labels']
    data_dict = {'labels': torch.LongTensor(labels)}

    if dataset_name == 'copa':
        choice1, choice2 = [x[0] for x in text_b], [x[1] for x in text_b]
        enc1 = tokenizer(text_a,
                         choice1,
                         padding='max_length',
                         max_length=max_seq_len,
                         truncation=True,
                         return_tensors='pt')
        enc2 = tokenizer(text_a,
                         choice2,
                         padding='max_length',
                         max_length=max_seq_len,
                         truncation=True,
                         return_tensors='pt')

        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            data_dict[key] = []
            for val1, val2 in zip(enc1[key], enc2[key]):
                data_dict[key].append(torch.stack((val1, val2)))
            data_dict[key] = torch.stack(data_dict[key])

    elif dataset_name == 'record':
        data_dict['index'] = data['index']
        data_dict['question_id'] = data['question_id']
        data_dict['entity'] = data['entity']
        data_dict['answers'] = data['answers']
        data_dict.update(
            tokenizer(text_a,
                      text_b,
                      padding='max_length',
                      max_length=max_seq_len,
                      truncation=True,
                      return_tensors='pt'))

    else:
        data_dict.update(
            tokenizer(text_a,
                      text_b,
                      padding='max_length',
                      max_length=max_seq_len,
                      truncation=True,
                      return_tensors='pt'))

    data_dict = DatasetDict(data_dict)
    return data_dict
