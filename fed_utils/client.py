import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
)
from .adaptive_peft import tokenize
import logging
import evaluate
import numpy as np


class GeneralClient:
    def __init__(self, client_id, model, tokenizer, prompter, data_path, output_dir, cutoff_len=512, train_on_inputs=True,
                 cache_dir = None, hetero_lora = False, optim = 'adamw_torch'):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.eval_data_path = os.path.join(data_path, "local_eval_{}.json".format(self.client_id))
        self.test_data_path = os.path.join(data_path, "local_test_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path, cache_dir = cache_dir)
        self.eval_data = load_dataset("json", data_files=self.eval_data_path, cache_dir = cache_dir)
        self.test_data = load_dataset("json", data_files=self.test_data_path, cache_dir=cache_dir)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.train_on_inputs = train_on_inputs
        self.cutoff_len = cutoff_len
        self.mask = None
        self.scheduler = None
        self.hetero_lora = hetero_lora
        self.optim = optim

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(self.tokenizer, full_prompt, cutoff_len=self.cutoff_len, add_eos_token=True)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    def preprare_local_dataset(self, local_val_set_size=0):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            self.local_eval_dataset = self.eval_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            self.local_test_dataset = self.test_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
        self.local_val_set_size = len(self.local_eval_dataset)

    def get_sparse(self, model, local_learning_rate, local_micro_batch_size, warmup, density):
        step = len(self.local_train_dataset) // local_micro_batch_size
        mask = create_sparse_mask(model, sparsity=density)  # Assuming this function creates the mask
        if self.optim == 'sgd':
            self.mask = Maskedsgd(model.parameters(), mask, lr=local_learning_rate)
        else:
            self.mask = MaskedAdam(model.parameters(), mask, lr=local_learning_rate)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.mask, num_warmup_steps=warmup, num_training_steps=step)

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            warmup=0,
                            density=None,
                            lambd=None,
                            reg = None):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        class reg_Trainer(transformers.Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                regularizer = 0
                count = 0
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        rank = min(param.data.shape[0], param.data.shape[1])
                        prune_rank = round(0.99 * rank)
                        if 'lora_A' in name:
                            prune_norm_A = torch.norm(param.data[prune_rank:, :])
                        else:
                            prune_norm_B = torch.norm(param.data[:, prune_rank:])
                        if count != 2:
                            count += 1
                        if count == 2:
                            regularizer += prune_norm_A.to('cpu') * prune_norm_B.to('cpu')
                            count = 0

                loss += lambd * regularizer
                return (loss, outputs) if return_outputs else loss
        def compute_metrics(pred):
            labels_ids = pred.label_ids
            labels_ids[labels_ids == -100] = 1829
            # pred_ids = pred.predictions
            pred_ids = np.argmax(pred.predictions, axis=-1)
            # all unnecessary tokens are removed
            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
            rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)
            return {
                'rouge1': round(rouge_output["rouge1"], 4),
                'rouge2': round(rouge_output["rouge2"], 4),
                'rougeL': round(rouge_output["rougeL"], 4),
                'rougeLsum': round(rouge_output["rougeLsum"], 4)
            }
        if ddp:
            gradient_accumulation_steps = gradient_accumulation_steps // world_size
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            do_train=True,
            do_eval=True,
            # fp16=True,
            logging_steps=1,
            optim=self.optim,
            # optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="no",
            output_dir=self.local_output_dir,
            # save_total_limit=1,
            # load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
        )
        if self.hetero_lora:
            if self.mask:
                self.local_trainer = reg_Trainer(model=self.model,
                                                 train_dataset=self.local_train_dataset,
                                                 eval_dataset=self.local_eval_dataset,
                                                 args=self.train_args,
                                                 data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                      padding=True
                                                 ),
                                                 optimizers=(self.mask, self.scheduler),
                                                 compute_metrics=compute_metrics
                                                 )
            else:
                self.local_trainer = reg_Trainer(model=self.model,
                                                 train_dataset=self.local_train_dataset,
                                                 eval_dataset=self.local_eval_dataset,
                                                 args=self.train_args,
                                                 data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                      padding=True
                                                 ),
                                                 compute_metrics=compute_metrics
                                                 )
        else:
            if self.mask:
                self.local_trainer = transformers.Trainer(model=self.model,
                                                          train_dataset=self.local_train_dataset,
                                                          eval_dataset=self.local_eval_dataset,
                                                          args=self.train_args,
                                                          data_collator=transformers.DataCollatorForSeq2Seq(
                                                              tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                              padding=True
                                                          ),
                                                          optimizers=(self.mask, self.scheduler),
                                                          compute_metrics=compute_metrics
                                                          )
            else:
                self.local_trainer = transformers.Trainer(model=self.model,
                                                          train_dataset=self.local_train_dataset,
                                                          eval_dataset=self.local_eval_dataset,
                                                          args=self.train_args,
                                                          data_collator=transformers.DataCollatorForSeq2Seq(
                                                              tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                          ),
                                                          compute_metrics=compute_metrics
                                                          )

    def initiate_local_training(self, sparse=False):
        self.model.config.use_cache = False
        if sparse:
            self.params_dict_old = copy.deepcopy(
                OrderedDict((name, param.detach()) for name, param in self.model.named_parameters()))
            self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters())
            # self.model.state_dict = (
            #     lambda instance, *_, **__: self.model.state_dict()
            # ).__get__(self.model, type(self.model))
        else:
            self.params_dict_old = copy.deepcopy(
                OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                            "lora" in name))
            self.params_dict_new = OrderedDict(
                (name, param.detach()) for name, param in self.model.named_parameters() if
                "lora" in name)
            self.model.state_dict = (
                lambda instance, *_, **__: get_peft_model_state_dict(
                    instance, self.params_dict_new, "lora"
                )
            ).__get__(self.model, type(self.model))

    def train(self):
        result = self.local_trainer.train()
        logging.info(self.local_trainer.state.log_history[-2])
        logging.info(self.local_trainer.state.log_history[-1])
        logging.info(result.metrics)

        return self.local_trainer.state.log_history[-2]

    # TODO: look at logits/LoRA weight difference (norm)
    def test(self, epoch, local_micro_batch_size):
        test_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_eval=True,
            # fp16=True,
            per_device_eval_batch_size=local_micro_batch_size,
            dataloader_drop_last=False,
            eval_accumulation_steps=4,
        )

        def compute_metrics(pred):
            labels_ids = pred.label_ids
            labels_ids[labels_ids == -100] = 1829
            # pred_ids = pred.predictions
            pred_ids = np.argmax(pred.predictions, axis=-1)
            # all unnecessary tokens are removed
            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
            rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)
            return {
                'rouge1': round(rouge_output["rouge1"], 4),
                'rouge2': round(rouge_output["rouge2"], 4),
                'rougeL': round(rouge_output["rougeL"], 4),
                'rougeLsum': round(rouge_output["rougeLsum"], 4)
            }

        # init trainer
        tester = transformers.Trainer(
            model=self.model,
            args=test_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            compute_metrics=compute_metrics
        )
        # test_dataset = self.test_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
        # test_dataset = self.local_test_dataset
        eval_dataset = self.local_eval_dataset
        # test_results = tester.evaluate(test_dataset)
        eval_results = tester.evaluate(eval_dataset)
        # logging.info('For client ' + str( self.client_id) + ', the test result is:')
        # logging.info(test_results)
        logging.info('For client ' + str(self.client_id) + ', the eval result is:')
        logging.info(eval_results)
        return eval_results

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        # new_adapter_weight = self.model.state_dict()
        if self.mask:
            single_output_dir = os.path.join(self.output_dir, str(self.client_id),
                                             "local_output_epoch_{}".format(epoch))
            os.makedirs(single_output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), single_output_dir + "/pytorch_model.bin")
        else:
            lora_params = {}
            for name, param in self.model.named_parameters():
                # if 'up_proj.lora' in name or 'down_proj.lora' in name or 'gate_proj.lora' in name:
                if 'lora' in name and param.requires_grad:
                    lora_params[name] = param
            single_output_dir = os.path.join(self.output_dir, str(self.client_id), "local_output_epoch_{}".format(epoch))
            os.makedirs(single_output_dir, exist_ok=True)
            torch.save(lora_params, single_output_dir + "/pytorch_model.bin")

        # older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "global")
        # set_peft_model_state_dict(self.model, older_adapter_weight, "global")

        _ = self.model.load_state_dict(self.params_dict_old, strict=False)
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

class MaskedAdam(torch.optim.AdamW):
    def __init__(self, params, mask, *args, **kwargs):
        super(MaskedAdam, self).__init__(params, *args, **kwargs)
        self.mask = mask

    def step(self, closure=None):
        # If a closure is provided, compute the loss
        loss = None
        if closure is not None:
            loss = closure()
        # Apply the mask before performing an optimization step
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    param_state = self.state[p]
                    # if 'step' in param_state and param_state['step'] >= 1024:
                    p.grad.data *= self.mask[i].to(p.grad.data.device)
                    self.mask[i].to('cpu')

        # Call the original step function
        return super(MaskedAdam, self).step(closure)

class Maskedsgd(torch.optim.SGD):
    def __init__(self, params, mask, *args, **kwargs):
        super(Maskedsgd, self).__init__(params, *args, **kwargs)
        self.mask = mask

    def step(self, closure=None):
        # If a closure is provided, compute the loss
        loss = None
        if closure is not None:
            loss = closure()
        # Apply the mask before performing an optimization step
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    param_state = self.state[p]
                    # if 'step' in param_state and param_state['step'] >= 1024:
                    p.grad.data *= self.mask[i].to(p.grad.data.device)
                    self.mask[i].to('cpu')

        # Call the original step function
        return super(Maskedsgd, self).step(closure)

def create_sparse_mask(model, sparsity=0.5):
    mask = []
    for name, param in model.named_parameters():
        param_mask = torch.rand(param.shape) < sparsity  # Random mask with specified sparsity
        mask.append(param_mask.float())
    return mask
