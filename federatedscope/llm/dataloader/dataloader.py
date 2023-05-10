from enum import Enum


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


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
                   target=['answers'],
                   tok_len=128):
    def get_text(list_of_dict):
        if len(list_of_dict) <= 0:
            return list_of_dict
        if isinstance(list_of_dict[0], str):
            return list_of_dict
        if isinstance(list_of_dict[0], dict):
            if 'text' in list_of_dict[0]:
                if isinstance(list_of_dict[0]['text'], list):
                    return [' '.join(x['text']) for x in list_of_dict]
                elif isinstance(list_of_dict[0]['text'], str):
                    return [x['text'] for x in list_of_dict]
            raise NotImplementedError

    rm_cols = dataset.column_names
    dataset_source = dataset.map(lambda batch: tokenizer(
        *[get_text(batch[k]) for k in source],
        truncation=True,
        return_tensors="pt",
        padding="longest",
        max_length=tok_len,
    ),
                                 batched=True,
                                 remove_columns=rm_cols)

    rm_cols = dataset.column_names
    dataset_target = dataset.map(lambda batch: tokenizer(
        *[get_text(batch[k]) for k in target],
        truncation=True,
        return_tensors="pt",
        padding="longest",
        max_length=tok_len,
    ),
                                 batched=True,
                                 remove_columns=rm_cols)
    return dataset_source, dataset_target


def load_llm_dataset(config=None):
    from datasets import load_dataset

    model_name, _ = config.model.type.split('@')
    dataset_name, _ = config.data.type.split('@')

    tokenizer = get_tokenizer(model_name,
                              cache_dir=config.data.root,
                              tok_len=config.llm.tok_len)

    dataset = load_dataset(dataset_name, cache_dir=config.data.root)
    print(dataset.column_names.keys())

    train_encoded_dataset = encode_dataset(dataset['train'],
                                           tokenizer,
                                           source=config.llm.dataset.source,
                                           target=config.llm.dataset.target,
                                           tok_len=config.llm.tok_len)
    val_encoded_dataset = encode_dataset(dataset['validation'],
                                         tokenizer,
                                         source=config.llm.dataset.source,
                                         target=config.llm.dataset.target,
                                         tok_len=config.llm.tok_len)
    return dict(train=train_encoded_dataset, val=val_encoded_dataset)


if __name__ == '__main__':
    # Test cases
    from federatedscope.core.configs.config import CN

    config = CN()
    config.model = CN()
    config.model.type = 'gpt2@huggingface_llm'

    config.llm = CN()
    config.llm.tok_len = 1000

    config.llm.dataset = CN()
    config.llm.dataset.source = ['question']
    config.llm.dataset.target = ['question', 'answers']

    config.data = CN()
    config.data.root = 'data'
    config.data.type = 'squad@llm'

    dataset = load_llm_dataset(config)
