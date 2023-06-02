import os
import json
import torch
import transformers

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, LLMDataset


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def get_tokenizer(model_name, cache_dir, tok_len=128):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=tok_len,
        padding_side="right",
        use_fast=False,
    )

    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens


def load_llm_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)

    # The data format is supposed to be a json file
    # Example: config.data.type: xxx.json@llm
    dataset_name, _ = config.data.type.split('@')
    fp = os.path.join(config.data.root, dataset_name)
    if dataset_name.endswith('.json'):
        with open(fp, 'r', encoding="utf-8") as f:
            list_data_dict = json.load(f)
    elif dataset_name.endswith('.jsonl'):
        list_data_dict = []
        with open(fp, 'r', encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if 'databricks-dolly-15k' in dataset_name:
                    new_item = dict(instruction=item['instruction'],
                                    input=item['context'],
                                    output=item['response'],
                                    category=item['category'])
                    item = new_item
                list_data_dict.append(item)
    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    dataset = LLMDataset(list_data_dict, tokenizer)

    return dataset, config
