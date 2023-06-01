from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer

# An example for converting torch training process to FS training process

# Refer to `federatedscope.core.trainers.BaseTrainer` for interface.


class HFTrainer(BaseTrainer):
    def __init__(self, model, data, device, **kwargs):
        import torch
        from transformers import Trainer, TrainingArguments
        from federatedscope.llm.dataloader import get_tokenizer, \
            LLMDataCollator

        self.model = model
        self.data = data
        self.device = device
        self.kwargs = kwargs
        self.criterion = torch.nn.CrossEntropyLoss()

        config = self.kwargs['config']
        self.cfg = config

        model_name, _ = config.model.type.split('@')
        tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                     config.llm.tok_len)
        data_collator = LLMDataCollator(tokenizer=tokenizer)
        training_args = \
            TrainingArguments(
                "hf-trainer",
                num_train_epochs=config.train.local_update_steps,
                report_to=None,
            )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data["train"].dataset,
            eval_dataset=data["train"].dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    def train(self):
        self.model.to(self.device)
        self.model.train()

        self.trainer.train()
        loss = self.trainer.state.log_history[-1]['loss']
        step = self.trainer.state.log_history[-1]['step']
        num_examples = step * self.cfg.dataloader.batch_size

        return num_examples, self.model.cpu().state_dict(), \
            {'loss_total': loss, 'avg_loss': loss/float(
                num_examples)}

    def evaluate(self, target_data_split_name='test'):
        self.model.to(self.device)
        self.model.eval()
        metric = self.trainer.evaluate(
            eval_dataset=self.data[target_data_split_name].dataset)
        print(metric)

        total_loss = 1.0
        num_samples = 2

        return {
            f'{target_data_split_name}_loss': total_loss,
            f'{target_data_split_name}_total': num_samples,
            f'{target_data_split_name}_avg_loss': total_loss /
            float(num_samples)
        }

    def update(self, model_parameters, strict=False):
        self.model.load_state_dict(model_parameters, strict)

    def get_model_para(self):
        return self.model.cpu().state_dict()


def call_hf_trainer(trainer_type):
    if trainer_type == 'hftrainer':
        return HFTrainer


register_trainer('hftrainer', call_hf_trainer)
