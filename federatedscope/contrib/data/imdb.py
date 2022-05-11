import os
import os.path as osp
import random
import logging
import torch
import urllib.request
import tarfile
from torch.utils.data.dataset import TensorDataset
from federatedscope.contrib.auxiliaries.utils import reporthook

logger = logging.getLogger(__name__)


def read_file(path):
    with open(path) as f:
        data = f.readline()
    return data


def create_imdb_examples(root, split):
    if not osp.exists(osp.join(root, split)):
        logger.info('Downloading imdb dataset')
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        data_dir = '/'.join(osp.normpath(root).split('/')[:-1])
        os.makedirs(data_dir, exist_ok=True)
        data_name = osp.normpath(root).split('/')[-1]
        data_file = osp.join(data_dir, '{}.zip'.format(data_name))
        urllib.request.urlretrieve(url, data_file, reporthook)
        with tarfile.open(data_file) as tar_ref:
            tar_ref.extractall(data_dir)
        os.remove(data_file)

    examples = []
    pos_files = os.listdir(osp.join(root, split, 'pos'))
    for file in pos_files:
        path = osp.join(root, split, 'pos', file)
        data = read_file(path)
        examples.append((data, 1))
    neg_files = os.listdir(osp.join(root, split, 'neg'))
    for file in neg_files:
        path = osp.join(root, split, 'neg', file)
        data = read_file(path)
        examples.append((data, 0))
    random.shuffle(examples)

    if split == 'train':
        num_train_samples = int(0.9 * len(examples))
        return examples[:num_train_samples], examples[num_train_samples:]
    elif split == 'test':
        return examples


def create_imdb_dataset(root, split, tokenizer, max_seq_len, model_type, cache_dir=''):
    logger.info('Preprocessing {} {} dataset'.format('imdb', split))
    cache_file = osp.join(cache_dir, 'imdb', '_'.join(['imdb', split, str(max_seq_len), model_type]) + '.pt')
    if osp.exists(cache_file):
        logger.info('Loading cache file from \'{}\''.format(cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        examples = create_imdb_examples(root, split)
        encoded_inputs = None

    def _create_dataset(examples_, encoded_inputs_=None):
        texts = [ex[0] for ex in examples_]
        labels = [ex[1] for ex in examples_]
        if encoded_inputs_ is None:
            encoded_inputs_ = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors='pt')

        dataset = TensorDataset(encoded_inputs_.input_ids, encoded_inputs_.token_type_ids,
                                encoded_inputs_.attention_mask, torch.LongTensor(labels))
        return dataset, encoded_inputs_, examples_

    if split == 'train':
        if encoded_inputs is not None:
            return _create_dataset(examples[0], encoded_inputs[0]), _create_dataset(examples[1], encoded_inputs[1])
        else:
            train_dataset, train_encoded, train_examples = _create_dataset(examples[0])
            val_dataset, val_encoded, val_examples = _create_dataset(examples[1])
            if cache_dir:
                logger.info('Saving cache file to \'{}\''.format(cache_file))
                os.makedirs(osp.join(cache_dir, 'imdb'), exist_ok=True)
                torch.save({'examples': examples,
                            'encoded_inputs': [train_encoded, val_encoded]}, cache_file)

        return (train_dataset, train_encoded, train_examples), (val_dataset, val_encoded, val_examples)

    elif split == 'test':
        if encoded_inputs is not None:
            return _create_dataset(examples, encoded_inputs)
        else:
            test_dataset, test_encoded, test_examples = _create_dataset(examples)
            if cache_dir:
                logger.info('Saving cache file to \'{}\''.format(cache_file))
                os.makedirs(osp.join(cache_dir, 'imdb'), exist_ok=True)
                torch.save({'examples': examples,
                            'encoded_inputs': test_encoded}, cache_file)

        return test_dataset, test_encoded, test_examples
