import torch
import logging
from torch import nn
from torch.nn import MSELoss
from lm_eval.base import BaseLM
from federatedscope.nlp.loss import SoftCrossEntropyLoss

logger = logging.getLogger(__name__)


class ModelOutput(object):
    def __init__(self,
                 loss=None,
                 regular_loss=None,
                 kd_loss=None,
                 logits=None,
                 hidden_states=None):
        self.loss = loss
        self.regular_loss = regular_loss
        self.kd_loss = kd_loss
        self.logits = logits
        self.hidden_states = hidden_states


class PLModel(nn.Module):
    def __init__(self, config, role):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.role = role
        self.only_use_hidden_loss = config.only_use_hidden_loss
        assert 'gpt' in config.model_type
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_type,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32)

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        self.n_head = self.model.config.num_attention_heads
        self.n_embd = self.hidden_size // self.n_head
        self.n_layer = config.get(f'num_{role}_layers')
        self.share_layer_param = config.get(f'share_{role}_layer_param')
        # if self.share_layer_param:
        #     num_layer_per_cell = config.get(f'num_{role}_layers_per_cell')
        #     assert self.n_layer % num_layer_per_cell == 0
        #     self.model.model.decoder.layers = nn.ModuleList(
        #         list(self.model.model.decoder.layers[:num_layer_per_cell]) *
        #         (self.n_layer // num_layer_per_cell))
        # else:
        #     self.model.model.decoder.layers = \
        #         self.model.model.decoder.layers[:self.n_layer]

        self.prefix_len = config.get(f'{role}_prefix_len')
        if self.prefix_len > 0:
            self.prefix_tokens = torch.arange(self.prefix_len).long()
            self.prefix_encoder = nn.Embedding(
                self.prefix_len,
                self.n_layer * 2 * self.hidden_size,
                dtype=torch.float16 if config.use_fp16 else torch.float32)

        self._freeze_param(config.get(f'{role}_freeze_param'))

    def generate(self, **kwargs):
        if self.prefix_len > 0:
            past_key_values = self._get_prompt(kwargs['batch_size'])
            kwargs['past_key_values'] = past_key_values
        return self.model.generate(**kwargs)

    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                use_kd_loss=False,
                teacher_model=None):
        if use_kd_loss:
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask,
                                                labels)

        batch_size = input_ids.size(0)
        past_key_values = None
        if attention_mask is None:
            attention_mask = torch.ones(batch_size,
                                        input_ids.size(1)).to(input_ids.device)
        if self.prefix_len > 0:
            past_key_values = self._get_prompt(batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(
                attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask),
                                       dim=1)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            # position_ids=torch.arange(input_ids.size(1)),
            past_key_values=past_key_values,
            labels=labels,
            output_hidden_states=True,
        )

        logits = outputs.logits
        loss = outputs.loss
        regular_loss, kd_loss = None, None
        if use_kd_loss:
            mse_loss_func, sce_loss_func = MSELoss(), SoftCrossEntropyLoss()
            hidden_loss = mse_loss_func(outputs.hidden_states[-1],
                                        teacher_outputs.hidden_states[-1])
            pred_loss = sce_loss_func(logits, teacher_outputs.logits)
            kd_loss = hidden_loss + pred_loss
            loss = hidden_loss if self.only_use_hidden_loss else \
                0.5 * loss + 0.5 * kd_loss

        return ModelOutput(
            loss=loss,
            regular_loss=regular_loss,
            kd_loss=kd_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def _get_prompt(self, batch_size):
        assert self.prefix_len > 0
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(
            batch_size, -1).to(self.model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd,
        )
        past_key_values = nn.functional.dropout(past_key_values,
                                                p=self.model.config.dropout,
                                                training=self.training)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def _freeze_param(self, params):
        for n, p in self.named_parameters():
            for fp in params:
                if fp in n:
                    p.requires_grad = False
                    break

    def _unfreeze_param(self, params):
        for n, p in self.named_parameters():
            for fp in params:
                if fp in n:
                    p.requires_grad = True
                    break


class LMEvalModel(BaseLM):
    def __init__(self, model, tokenizer, device, batch_size=1):
        super().__init__()
        assert isinstance(batch_size, int)
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self._device = device
        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if hasattr(self.model.model.config, 'n_ctx'):
            return self.model.model.config.n_ctx
        elif hasattr(self.model.model.config, 'max_position_embeddings'):
            return self.model.model.config.max_position_embeddings
        else:
            return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            out = self.model(inps).logits
            return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False,
            batch_size=self.batch_size,
        )
