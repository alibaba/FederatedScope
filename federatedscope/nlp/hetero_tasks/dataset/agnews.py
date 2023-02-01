import os
import os.path as osp
import logging
import torch
from federatedscope.nlp.hetero_tasks.dataset.utils import split_sent, \
    DatasetDict, NUM_DEBUG

logger = logging.getLogger(__name__)


def get_agnews_examples(data, is_debug=False):
    if is_debug:
        data = data[:NUM_DEBUG]
    examples = []
    for ex in data:
        examples.append((ex['text'], ex['label']))
    return examples


def process_agnews_dataset(data,
                           split,
                           tokenizer,
                           max_seq_len,
                           cache_dir='',
                           client_id=None,
                           pretrain=False,
                           is_debug=False,
                           **kwargs):
    if pretrain:
        return process_agnews_dataset_for_pretrain(data, split, tokenizer,
                                                   max_seq_len, cache_dir,
                                                   client_id, is_debug)

    save_dir = osp.join(cache_dir, 'train', str(client_id))
    cache_file = osp.join(save_dir, split + '.pt')
    if osp.exists(cache_file):
        logger.info('Loading cache file from \'{}\''.format(cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        examples = get_agnews_examples(data, is_debug)
        texts = [ex[0] for ex in examples]
        encoded_inputs = tokenizer(texts,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=max_seq_len,
                                   return_tensors='pt')

        if cache_dir:
            logger.info('Saving cache file to \'{}\''.format(cache_file))
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'examples': examples,
                'encoded_inputs': encoded_inputs
            }, cache_file)

    labels = [ex[1] for ex in examples]
    example_indices = torch.arange(encoded_inputs.input_ids.size(0),
                                   dtype=torch.long)
    dataset = DatasetDict({
        'token_ids': encoded_inputs.input_ids,
        'token_type_ids': encoded_inputs.token_type_ids,
        'attention_mask': encoded_inputs.attention_mask,
        'labels': torch.LongTensor(labels),
        'example_indices': example_indices
    })
    return dataset, encoded_inputs, examples


def process_agnews_dataset_for_pretrain(data,
                                        split,
                                        tokenizer,
                                        max_seq_len,
                                        cache_dir='',
                                        client_id=None,
                                        is_debug=False):
    save_dir = osp.join(cache_dir, 'pretrain', str(client_id))
    cache_file = osp.join(save_dir, split + '.pt')

    if osp.exists(cache_file):
        logger.info('Loading cache file from \'{}\''.format(cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        examples = get_agnews_examples(data, is_debug)
        texts = [ex[0] for ex in examples]
        texts = split_sent(texts, eoq=tokenizer.eoq_token)
        encoded_inputs = tokenizer(texts,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=max_seq_len,
                                   return_tensors='pt')
        num_non_padding = (encoded_inputs.input_ids !=
                           tokenizer.pad_token_id).sum(dim=-1)
        for i, pad_idx in enumerate(num_non_padding):
            encoded_inputs.input_ids[i, 0] = tokenizer.bos_token_id
            encoded_inputs.input_ids[i, pad_idx - 1] = tokenizer.eos_token_id

        if cache_dir:
            logger.info('Saving cache file to \'{}\''.format(cache_file))
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'examples': examples,
                'encoded_inputs': encoded_inputs
            }, cache_file)

    example_indices = torch.arange(encoded_inputs.input_ids.size(0),
                                   dtype=torch.long)
    dataset = DatasetDict({
        'token_ids': encoded_inputs.input_ids,
        'attention_mask': encoded_inputs.attention_mask,
        'example_indices': example_indices
    })
    return dataset, encoded_inputs, examples
