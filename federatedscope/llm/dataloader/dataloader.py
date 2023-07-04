import os
import gzip
import json
import random
import logging
import torch
import transformers

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, LLMDataset
from federatedscope.core.data.utils import download_url

logger = logging.getLogger(__name__)


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


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category'):
    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_llm_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)

    dataset_name, _ = config.data.type.split('@')

    if dataset_name.endswith('.json'):
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.endswith('.jsonl'):
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_jsonl(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'alpaca':
        fp = os.path.join(config.data.root, 'alpaca_data.json')
        download_url(
            'https://raw.githubusercontent.com/tatsu-lab'
            '/stanford_alpaca/'
            '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
            'alpaca_data.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'alpaca_cleaned':
        fp = os.path.join(config.data.root, 'alpaca_data_cleaned.json')
        download_url(
            'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/'
            'a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/'
            'alpaca_data_cleaned.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'dolly-15k':
        fp = os.path.join(config.data.root, 'databricks-dolly-15k.jsonl')
        download_url(
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl', config.data.root)
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='context',
                                    output='response',
                                    category='category')
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'gsm8k':
        fp = os.path.join(config.data.root, 'gsm8k_train.jsonl')
        if not os.path.exists(fp):
            download_url(
                'https://raw.githubusercontent.com/openai/grade-school-math'
                '/3101c7d5072418e28b9008a6636bde82a006892c/'
                'grade_school_math/data/train.jsonl', config.data.root)
            os.rename(os.path.join(config.data.root, 'train.jsonl'), fp)
        list_data_dict = load_jsonl(fp,
                                    instruction='question',
                                    output='answer')
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('####', 'The answer is')
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'code_search_net':
        from tqdm import tqdm
        from federatedscope.llm.dataset.code_search_net import \
            CSN_FILE_NUM_DICT

        list_data_dict = []
        logger.info('Loading code search net data file...')
        try:
            for language in tqdm(CSN_FILE_NUM_DICT.keys()):
                sub_list_data_dict = []
                for file_index in range(CSN_FILE_NUM_DICT[language]['train']):
                    fp = \
                        os.path.join(config.data.root, language,
                                     'final', 'jsonl', 'train',
                                     f'{language}_train_{file_index}.jsonl.gz')
                    tmp_list_data_dict = load_jsonl(
                        fp,
                        instruction='docstring',
                        input='language',
                        output='code',
                        category='language',
                        is_gzip=True,
                    )
                    sub_list_data_dict += tmp_list_data_dict
                # Subsample
                raw_size = len(sub_list_data_dict)
                num_subsample = int(raw_size * config.data.subsample)
                list_data_dict += random.sample(sub_list_data_dict,
                                                num_subsample)
                logger.info(f"Subsample "
                            f"{sub_list_data_dict[0]['category']} with "
                            f"rate {config.data.subsample}: "
                            f"the sample size is # {num_subsample} "
                            f"(the raw size is {raw_size}).")
            # Modify instruction with specific language
            for sample in list_data_dict:
                sample['instruction'] = \
                    sample['category'] + ' ' + sample['instruction']
        except FileNotFoundError:
            raise FileNotFoundError(
                'Data not found! Please run `python '
                'federatedscope/llm/dataset/code_search_net.py` '
                'to download data.')
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'rosetta_alpaca':
        fp = os.path.join(config.data.root, 'rosetta_alpaca.json')
        download_url(
            'https://raw.githubusercontent.com/'
            'sahil280114/codealpaca/'
            'd269da106a579a623a654529b3cb91b5dfa9c72f/'
            'data/rosetta_alpaca.json', config.data.root)
        list_data_dict = load_json(fp,
                                   instruction='instruction',
                                   input='input',
                                   output='output',
                                   category='input')
        # Remove 'x86-64 Assembl' if splitter is `meta` due to the number of
        # samples is too small.
        if config.data.splitter == 'meta':
            list_data_dict = [
                i for i in list_data_dict if i['category'] != 'X86-64 Assembly'
            ]
        dataset = LLMDataset(list_data_dict, tokenizer)
    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    return dataset, config
