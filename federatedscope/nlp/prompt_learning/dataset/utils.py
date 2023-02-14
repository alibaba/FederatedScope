from torch.utils.data.dataset import Dataset

NUM_DEBUG = 20


class DatasetDict(Dataset):
    def __init__(self, inputs):
        super().__init__()
        assert all(
            len(list(inputs.values())[0]) == len(v)
            for v in inputs.values()), "Size mismatch between tensors"
        self.inputs = inputs

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.inputs.items()}

    def __len__(self):
        return len(list(self.inputs.values())[0])


def setup_tokenizer(config):
    from transformers.models.bert import BertTokenizerFast
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            config.model.model_type,
            skip_special_tokens=True,
            local_files_only=True,
        )
    except:
        tokenizer = BertTokenizerFast.from_pretrained(
            config.model.model_type,
            skip_special_tokens=True,
        )
    return tokenizer
