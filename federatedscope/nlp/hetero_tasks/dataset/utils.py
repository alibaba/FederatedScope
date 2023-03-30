import os
import json
import numpy as np
import logging

try:
    import torch
    from torch.utils.data.dataset import Dataset
except ImportError:
    torch = None
    Dataset = None

NUM_DEBUG = 20
BOS_TOKEN_ID = -1
EOS_TOKEN_ID = -1
EOQ_TOKEN_ID = -1
PAD_TOKEN_ID = -1

logger = logging.getLogger(__name__)


def split_sent(examples, eoq='[unused2]', tokenize=True):
    import nltk
    from nltk.tokenize import sent_tokenize

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    new_examples = []
    for e in examples:
        if tokenize:
            e = f' {eoq} '.join(sent_tokenize(e))
        else:
            e = e.replace('[SEP]', eoq)
        new_examples.append(e)
    return new_examples


class DatasetDict(Dataset):
    def __init__(self, inputs):
        super().__init__()
        assert all(
            list(inputs.values())[0].size(0) == v.size(0)
            for v in inputs.values()), "Size mismatch between tensors"
        self.inputs = inputs

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.inputs.items()}

    def __len__(self):
        return list(self.inputs.values())[0].size(0)


def setup_tokenizer(model_type,
                    bos_token='[unused0]',
                    eos_token='[unused1]',
                    eoq_token='[unused2]'):
    """
    Get a tokenizer, the default bos/eos/eoq token is used for Bert
    """
    from transformers.models.bert import BertTokenizerFast
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            model_type,
            additional_special_tokens=[bos_token, eos_token, eoq_token],
            skip_special_tokens=True,
            local_files_only=True,
        )
    except:
        tokenizer = BertTokenizerFast.from_pretrained(
            model_type,
            additional_special_tokens=[bos_token, eos_token, eoq_token],
            skip_special_tokens=True,
        )

    tokenizer.bos_token = bos_token
    tokenizer.eos_token = eos_token
    tokenizer.eoq_token = eoq_token
    tokenizer.bos_token_id = tokenizer.vocab[bos_token]
    tokenizer.eos_token_id = tokenizer.vocab[eos_token]
    tokenizer.eoq_token_id = tokenizer.vocab[eoq_token]

    global BOS_TOKEN_ID, EOS_TOKEN_ID, EOQ_TOKEN_ID, PAD_TOKEN_ID
    BOS_TOKEN_ID = tokenizer.bos_token_id
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOQ_TOKEN_ID = tokenizer.eoq_token_id
    PAD_TOKEN_ID = tokenizer.pad_token_id

    return tokenizer


def load_synth_data(data_config):
    """
    Load the synthetic data for contrastive learning
    """
    if data_config.is_debug:
        synth_dir = 'cache_debug/synthetic/'
    else:
        synth_dir = os.path.join(data_config.cache_dir, 'synthetic')

    logger.info('Loading synthetic data from \'{}\''.format(synth_dir))
    synth_prim_weight = data_config.hetero_synth_prim_weight
    with open(os.path.join(synth_dir, 'shapes.json')) as f:
        shapes = json.load(f)
    synth_feat_path = os.path.join(
        synth_dir, 'feature_{}.memmap'.format(synth_prim_weight))
    synth_tok_path = os.path.join(synth_dir,
                                  'token_{}.memmap'.format(synth_prim_weight))
    synth_feats = np.memmap(filename=synth_feat_path,
                            shape=tuple(shapes['feature']),
                            mode='r',
                            dtype=np.float32)
    synth_toks = np.memmap(filename=synth_tok_path,
                           shape=tuple(shapes['token']),
                           mode='r',
                           dtype=np.int64)
    num_contrast = data_config.num_contrast
    synth_feats = {
        k: v
        for k, v in enumerate(torch.from_numpy(synth_feats)[:num_contrast])
    }
    synth_toks = {
        k: v
        for k, v in enumerate(torch.from_numpy(synth_toks)[:num_contrast])
    }

    return synth_feats, synth_toks
