from torch.utils.data.dataset import Dataset

NUM_DEBUG = 20


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


def setup_tokenizer(config):
    from transformers.models.bert import BertTokenizerFast

    bos_token, eos_token, eoq_token = \
        config.model.bos_token, config.model.eos_token, config.model.eoq_token
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            config.model.model_type,
            additional_special_tokens=[bos_token, eos_token, eoq_token],
            skip_special_tokens=True,
            local_files_only=True,
        )
    except:
        tokenizer = BertTokenizerFast.from_pretrained(
            config.model.model_type,
            additional_special_tokens=[bos_token, eos_token, eoq_token],
            skip_special_tokens=True,
        )
    tokenizer.bos_token = bos_token
    tokenizer.eos_token = eos_token
    tokenizer.eoq_token = eoq_token
    tokenizer.bos_token_id = tokenizer.vocab[bos_token]
    tokenizer.eos_token_id = tokenizer.vocab[eos_token]
    tokenizer.eoq_token_id = tokenizer.vocab[eoq_token]
    return tokenizer
