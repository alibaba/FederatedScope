import os
import json
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import copy
import numpy as np
from dataclasses import dataclass
import transformers
import torch


IGNORE_INDEX = -100


class LLMDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 use_prompts,
                 generation=False):
        super(LLMDataset, self).__init__()
        
        if use_prompts:
            # prompt template from alpaca
            sources = [f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example[0]}\n\n### Input:\n{example[1]}\n\n### Response:' for example in data]
        else:
            sources = [f'{example[0]}\n\nInput: {example[1]}\n\nOutput:' for example in data]
        targets = [f'{example[2]}{tokenizer.eos_token}' for example in data]

        data_dict = self.preprocess(sources, targets, tokenizer, generation)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer, generation):
        if generation:
            sources_tokenized, labels_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (sources, targets)
            ]
            input_ids = self._tokenize_fn(sources, tokenizer)["input_ids"]
            labels = self._tokenize_fn(targets, tokenizer)["input_ids"]
        else:
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (examples, sources)
            ]
            input_ids = examples_tokenized["input_ids"]
            labels = copy.deepcopy(input_ids)
            for label, source_len in zip(labels,
                                        sources_tokenized["input_ids_lens"]):
                label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i])


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
            padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        

def _get_task_splits():
    with open(os.path.join('data', 'natural-instructions-2.8', 'splits', 'default', 'train_tasks.txt'), 'r') as reader:
        train_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
    with open(os.path.join('data', 'natural-instructions-2.8', 'splits', 'default', 'test_tasks.txt'), 'r') as reader:
        eval_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
    return train_set_names, eval_set_names


def _filter_out_over_length(items, max_length):
    return [item for item in items if len(item['input']) < max_length]


def get_instruction_dataset(args, tokenizer, only_eval=False):
    """
    only_eval: only effective with zeroshot set to `True`
    """
    train_set_names, eval_set_names = _get_task_splits()
    list_train_loader = []
    data_collator = LLMDataCollator(tokenizer=tokenizer)
    
    # if only_eval, the following lines won't be executed to save time.
    if not only_eval:
        print('load train sets')
        for file_name in train_set_names:
            with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'tasks', file_name)) as reader:
                raw_data = json.load(reader)
                instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
                if len(instances) < 20:
                    continue
                # sample 20% dataset
                instances = np.random.choice(instances, int(len(instances) * 0.2), replace=False)
                print(file_name, len(instances), max([len(item['input']) for item in instances]))
                instruct = raw_data['Definition'][0]
                data = []
                for item in instances:
                    # only take the first output into consideration
                    data.append((instruct, item['input'], item['output'][0]))
                dataset = LLMDataset(data, tokenizer, use_prompts=args.use_prompts)
                list_train_loader.append(DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator))
        args.num_clients = len(list_train_loader)

    list_eval_set = []
    for file_name in eval_set_names:
        with open(os.path.join(os.path.expanduser('~'), '.datasets', 'natural-instructions-2.8', 'tasks', file_name)) as reader:
            raw_data = json.load(reader)
            instruct = raw_data['Definition'][0]
            instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
            if len(instances) > 20:
                # sample 2% instances
                instances = np.random.choice(instances, max(20, int(0.02 * len(instances))), replace=False)
            data = []
            for item in instances:
                # only take the first output into consideration
                data.append((instruct, item['input'], item['output'][0]))
            if args.eval_metric == 'loss':
                list_eval_set.append(LLMDataset(data, tokenizer, use_prompts=args.use_prompts, generation=False))
            else:
                list_eval_set.append(LLMDataset(data, tokenizer, use_prompts=args.use_prompts, generation=True))
    universal_eval_set = ConcatDataset(list_eval_set)
    eval_loader = DataLoader(universal_eval_set, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    return list_train_loader, eval_loader
