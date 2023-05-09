from enum import Enum


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"


def get_tokenizer(model_name, cache_dir, tok_len=128):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=tok_len,
        padding_side="right",
        use_fast=False,
    )

    special_tokens = {
        'pad_token': DefaultToken.PAD_TOKEN.value,
        'eos_token': DefaultToken.EOS_TOKEN.value,
        'bos_token': DefaultToken.BOS_TOKEN.value,
        'unk_token': DefaultToken.UNK_TOKEN.value,
    }
    tokenizer.add_special_tokens(special_tokens)

    return tokenizer


def encode_dataset(dataset,
                   tokenizer,
                   source=['question'],
                   target=['answers']):

    dataset_source = dataset.map(
        lambda batch: tokenizer(
            *[batch[k] for k in source],
            truncation=True,
        ),
        batched=True,
    )

    dataset_target = dataset.map(
        lambda batch: tokenizer(
            *[batch[k] for k in source],
            truncation=True,
        ),
        batched=True,
    )
    return dataset_source, dataset_target


def load_llm_dataset(config=None):
    from datasets import load_dataset

    model_name, _ = config.model.type.split('@')
    dataset_name, _ = config.data.type.split('@')

    tokenizer = get_tokenizer(model_name,
                              cache_dir=config.data.root,
                              tok_len=config.llm.tok_len)

    dataset = load_dataset(dataset_name, cache_dir=config.data.root)
    encoded_dataset = encode_dataset(dataset, tokenizer)

    return encoded_dataset


if __name__ == '__main__':
    # Test cases
    from federatedscope.core.configs.config import CN

    config = CN()
    config.model = CN()
    config.model.type = 'gpt2@huggingface_llm'

    config.llm = CN()
    config.llm.tok_len = 1000

    config.data = CN()
    config.data.root = 'data'
    config.data.type = 'squad@llm'

    dataset = load_llm_dataset(config)

    print(dataset)
