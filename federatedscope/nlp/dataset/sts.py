import os
import os.path as osp
import csv
import logging
import torch
import urllib.request
import zipfile
from torch.utils.data.dataset import TensorDataset
from federatedscope.nlp.auxiliaries.utils import reporthook

logger = logging.getLogger(__name__)


class GlueExample(object):
    def __init__(self, text_a, text_b, label):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        data = []
        for line in reader:
            data.append(line)
        return data


def create_sts_examples(root, split):
    data_path = osp.join(root, split + '.tsv')

    if not osp.exists(data_path):
        logger.info('Downloading sts dataset')
        url = 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip'
        data_dir = '/'.join(osp.normpath(root).split('/')[:-1])
        os.makedirs(data_dir, exist_ok=True)
        data_name = osp.normpath(root).split('/')[-1]
        data_file = osp.join(data_dir, '{}.zip'.format(data_name))
        urllib.request.urlretrieve(url, data_file, reporthook)
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(data_file)

    text_a_id, text_b_id, label_id = 7, 8, -1
    data = read_tsv(data_path)
    examples = []
    for i, line in enumerate(data):
        if i == 0: continue
        text_a = line[text_a_id]
        text_b = line[text_b_id]
        label = float(line[label_id])
        examples.append(GlueExample(text_a, text_b, label))

    if split == 'train':
        num_train_samples = int(0.9 * len(examples))
        return examples[:num_train_samples], examples[num_train_samples:]
    elif split == 'dev':
        return examples


def create_sts_dataset(root, split, tokenizer, max_seq_len, model_type, cache_dir=''):
    logger.info('Preprocessing {} {} dataset'.format('sts', split))
    cache_file = osp.join(cache_dir, 'sts', '_'.join(['sts', split, str(max_seq_len), model_type]) + '.pt')
    if osp.exists(cache_file):
        logger.info('Loading cache file from \'{}\''.format(cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        examples = create_sts_examples(root, split)
        encoded_inputs = None

    def _create_dataset(examples_, encoded_inputs_=None):
        texts_a = [ex.text_a for ex in examples_]
        texts_b = [ex.text_b for ex in examples_]
        labels = [ex.label for ex in examples_]

        if encoded_inputs_ is None:
            encoded_inputs_ = tokenizer(texts_a, texts_b, padding=True, truncation=True, max_length=max_seq_len, return_tensors='pt')

        dataset = TensorDataset(encoded_inputs_.input_ids, encoded_inputs_.token_type_ids,
                                encoded_inputs_.attention_mask, torch.FloatTensor(labels))
        return dataset, encoded_inputs_, examples_

    if split == 'train':
        if encoded_inputs is not None:
            return _create_dataset(examples[0], encoded_inputs[0]), _create_dataset(examples[1], encoded_inputs[1])
        else:
            train_dataset, train_encoded, train_examples = _create_dataset(examples[0])
            val_dataset, val_encoded, val_examples = _create_dataset(examples[1])
            if cache_dir:
                logger.info('Saving cache file to \'{}\''.format(cache_file))
                os.makedirs(osp.join(cache_dir, 'sts'), exist_ok=True)
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
                os.makedirs(osp.join(cache_dir, 'sts'), exist_ok=True)
                torch.save({'examples': examples,
                            'encoded_inputs': test_encoded}, cache_file)

        return test_dataset, test_encoded, test_examples
