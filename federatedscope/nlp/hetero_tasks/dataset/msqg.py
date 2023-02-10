import os
import os.path as osp
import logging
import torch
import numpy as np
from federatedscope.nlp.hetero_tasks.dataset.utils import split_sent, \
    DatasetDict, NUM_DEBUG

logger = logging.getLogger(__name__)


def get_msqg_examples(data, is_debug=False):
    if is_debug:
        data = data[:NUM_DEBUG]
    src_examples, tgt_examples = [], []
    for ex in data:
        src_examples.append(ex['src'])
        tgt_examples.append(ex['tgt'])
    return src_examples, tgt_examples


def process_msqg_dataset(data,
                         split,
                         tokenizer,
                         max_src_len,
                         max_tgt_len,
                         raw_cache_dir='',
                         client_id=None,
                         pretrain=False,
                         is_debug=False,
                         **kwargs):
    if pretrain:
        return process_msqg_dataset_for_pretrain(data, split, tokenizer,
                                                 max_src_len, raw_cache_dir,
                                                 client_id, is_debug)

    cache_dir = osp.join(raw_cache_dir, 'train', str(client_id), split)
    src_examples, tgt_examples = get_msqg_examples(data, is_debug)
    if osp.exists(cache_dir):
        logger.info('Loading cache file from \'{}\''.format(cache_dir))
        token_ids = np.memmap(filename=osp.join(cache_dir, 'token_ids.memmap'),
                              shape=(len(src_examples), max_src_len),
                              mode='r',
                              dtype=np.int64)
        token_type_ids = np.memmap(filename=osp.join(cache_dir,
                                                     'token_type_ids.memmap'),
                                   shape=(len(src_examples), max_src_len),
                                   mode='r',
                                   dtype=np.int64)
        attention_mask = np.memmap(filename=osp.join(cache_dir,
                                                     'attention_mask.memmap'),
                                   shape=(len(src_examples), max_src_len),
                                   mode='r',
                                   dtype=np.int64)
        labels = np.memmap(filename=osp.join(cache_dir, 'labels.memmap'),
                           shape=(len(src_examples), max_tgt_len),
                           mode='r',
                           dtype=np.int64)

        token_ids = torch.from_numpy(token_ids)
        token_type_ids = torch.from_numpy(token_type_ids)
        attention_mask = torch.from_numpy(attention_mask)
        labels = torch.from_numpy(labels)
    else:
        src_encoded = tokenizer(src_examples,
                                padding='max_length',
                                truncation=True,
                                max_length=max_src_len,
                                return_tensors='pt')
        tgt_examples = split_sent(tgt_examples,
                                  eoq=tokenizer.eoq_token,
                                  tokenize=False)
        tgt_encoded = tokenizer(tgt_examples,
                                padding='max_length',
                                truncation=True,
                                max_length=max_tgt_len,
                                return_tensors='pt')
        num_non_padding = (tgt_encoded.input_ids !=
                           tokenizer.pad_token_id).sum(dim=-1)
        for i, pad_idx in enumerate(num_non_padding):
            tgt_encoded.input_ids[i, 0] = tokenizer.bos_token_id
            tgt_encoded.input_ids[i, pad_idx - 1] = tokenizer.eos_token_id

        if raw_cache_dir:
            logger.info('Saving cache file to \'{}\''.format(cache_dir))
            os.makedirs(cache_dir, exist_ok=True)
            token_ids = np.memmap(filename=osp.join(cache_dir,
                                                    'token_ids.memmap'),
                                  shape=(len(src_examples), max_src_len),
                                  mode='w+',
                                  dtype=np.int64)
            token_type_ids = np.memmap(filename=osp.join(
                cache_dir, 'token_type_ids.memmap'),
                                       shape=(len(src_examples), max_src_len),
                                       mode='w+',
                                       dtype=np.int64)
            attention_mask = np.memmap(filename=osp.join(
                cache_dir, 'attention_mask.memmap'),
                                       shape=(len(src_examples), max_src_len),
                                       mode='w+',
                                       dtype=np.int64)
            labels = np.memmap(filename=osp.join(cache_dir, 'labels.memmap'),
                               shape=(len(src_examples), max_tgt_len),
                               mode='w+',
                               dtype=np.int64)

            for i in range(len(src_examples)):
                token_ids[i] = src_encoded.input_ids[i]
                token_type_ids[i] = src_encoded.token_type_ids[i]
                attention_mask[i] = src_encoded.attention_mask[i]
                labels[i] = tgt_encoded.input_ids[i]

            token_ids = torch.from_numpy(token_ids)
            token_type_ids = torch.from_numpy(token_type_ids)
            attention_mask = torch.from_numpy(attention_mask)
            labels = torch.from_numpy(labels)

        else:
            token_ids = src_encoded.input_ids
            token_type_ids = src_encoded.token_type_ids
            attention_mask = src_encoded.attention_mask
            labels = tgt_encoded.input_ids

    example_indices = torch.arange(token_ids.size(0), dtype=torch.long)
    dataset = DatasetDict({
        'token_ids': token_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'example_indices': example_indices
    })
    return dataset, None, None


def process_msqg_dataset_for_pretrain(data,
                                      split,
                                      tokenizer,
                                      max_src_len,
                                      raw_cache_dir='',
                                      client_id=None,
                                      is_debug=False):
    cache_dir = osp.join(raw_cache_dir, 'pretrain', str(client_id), split)
    src_examples, tgt_examples = get_msqg_examples(data, is_debug)
    if osp.exists(cache_dir):
        logger.info('Loading cache file from \'{}\''.format(cache_dir))
        token_ids = np.memmap(filename=osp.join(cache_dir, 'token_ids.memmap'),
                              shape=(len(src_examples), max_src_len),
                              mode='r',
                              dtype=np.int64)
        attention_mask = np.memmap(filename=osp.join(cache_dir,
                                                     'attention_mask.memmap'),
                                   shape=(len(src_examples), max_src_len),
                                   mode='r',
                                   dtype=np.int64)
        token_ids = torch.from_numpy(token_ids)
        attention_mask = torch.from_numpy(attention_mask)
    else:
        src_examples = split_sent(src_examples,
                                  eoq=tokenizer.eoq_token,
                                  tokenize=False)
        src_encoded = tokenizer(src_examples,
                                padding='max_length',
                                truncation=True,
                                max_length=max_src_len,
                                return_tensors='pt')
        num_non_padding = (src_encoded.input_ids !=
                           tokenizer.pad_token_id).sum(dim=-1)
        for i, pad_idx in enumerate(num_non_padding):
            src_encoded.input_ids[i, 0] = tokenizer.bos_token_id
            src_encoded.input_ids[i, pad_idx - 1] = tokenizer.eos_token_id

        if raw_cache_dir:
            logger.info('Saving cache file to \'{}\''.format(cache_dir))
            os.makedirs(cache_dir, exist_ok=True)
            token_ids = np.memmap(filename=osp.join(cache_dir,
                                                    'token_ids.memmap'),
                                  shape=(len(src_examples), max_src_len),
                                  mode='w+',
                                  dtype=np.int64)
            attention_mask = np.memmap(filename=osp.join(
                cache_dir, 'attention_mask.memmap'),
                                       shape=(len(src_examples), max_src_len),
                                       mode='w+',
                                       dtype=np.int64)

            for i in range(len(src_examples)):
                token_ids[i] = src_encoded.input_ids[i]
                attention_mask[i] = src_encoded.attention_mask[i]

            token_ids = torch.from_numpy(token_ids)
            attention_mask = torch.from_numpy(attention_mask)
        else:
            token_ids = src_encoded.input_ids
            attention_mask = src_encoded.attention_mask

    example_indices = torch.arange(token_ids.size(0), dtype=torch.long)
    dataset = DatasetDict({
        'token_ids': token_ids,
        'attention_mask': attention_mask,
        'example_indices': example_indices
    })
    return dataset, None, None
