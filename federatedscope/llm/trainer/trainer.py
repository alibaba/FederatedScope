from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer


class LLMTrainer(GeneralTorchTrainer):
    """
    TODO: implement this
    """
    pass


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)

if __name__ == '__main__':
    # Test cases
    import transformers
    from federatedscope.core.configs.config import CN
    from federatedscope.llm.dataloader.dataloader import load_llm_dataset
    from federatedscope.llm.model.model_builder import \
        get_model_from_huggingface, enable_adapter

    config = CN()
    config.seed = 42

    config.model = CN()
    config.model.type = 'gpt2@huggingface_llm'

    config.llm = CN()
    config.llm.tok_len = 1000

    config.llm.dataset = CN()
    config.llm.dataset.source = ['question']
    config.llm.dataset.target = ['question', 'answers']

    config.data = CN()
    config.data.root = 'data'
    config.data.type = 'alpaca_data.json@llm'
    config.data.splits = [0, 0.5, 0.5]

    dataset, data_collator, tokenizer, num_new_tokens = \
        load_llm_dataset(config)

    model = get_model_from_huggingface(model_name='gpt2',
                                       llm_config=config.llm)

    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    import torch
    import numpy as np

    # TODO: make trainer compatible fs_trainer
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(dataset,
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=data_collator,
                                  drop_last=True)

    epochs = 5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    losses = []
    model.train()

    model = enable_adapter(model, 'lora', 'adapterhub')
    model.train_adapter(["lora_adapter"])

    model.to('cuda:0')
    for i in range(epochs):
        for batch_idx, item in enumerate(train_dataloader):
            input_ids = item['input_ids'].to('cuda:0')
            labels = item['labels'].to('cuda:0')
            optimizer.zero_grad()
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            losses.append(loss.mean().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            if batch_idx % 1000 == 0:
                print(np.mean(losses))
