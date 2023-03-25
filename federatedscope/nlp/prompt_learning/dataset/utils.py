import torch
from torch.utils.data.dataset import Dataset

NUM_DEBUG = 20


class DatasetDict(Dataset):
    def __init__(self, inputs):
        super().__init__()
        assert all(
            len(list(inputs.values())[0]) == len(v)
            for v in inputs.values()), "Size mismatch between tensors"
        self.inputs = {
            k: torch.stack([torch.tensor(x) for x in v])
            for k, v in inputs.items()
        }

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.inputs.items()}

    def __len__(self):
        return len(list(self.inputs.values())[0])


def setup_tokenizer(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_type)
    if 'gpt2' in config.model.model_type:
        tokenizer.pad_token = tokenizer.bos_token

    return tokenizer


def map_dataset_name_and_config(dataset_name):
    dataset_config_name = None
    if dataset_name == 'arc_easy':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Easy'
    elif dataset_name == 'arc_challenge':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Challenge'
    elif dataset_name == 'race':
        dataset_config_name = 'high'
    elif dataset_name == 'wikitext':
        dataset_config_name = 'wikitext-2-raw-v1'

    return dataset_name, dataset_config_name
