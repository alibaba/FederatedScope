import torch
import logging
from torch import nn
from torch.nn import MSELoss
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
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


class MyOPTLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, dtype=None):
        self.offset = 2
        super().__init__(num_embeddings + self.offset,
                         embedding_dim,
                         dtype=dtype)

    def forward(self, attention_mask, past_key_values_length):
        attention_mask = attention_mask.long()
        positions = torch.arange(
            attention_mask.size(1) -
            past_key_values_length).type_as(attention_mask)

        return super().forward(positions + self.offset)


class MyGPT2Attention(GPT2Attention):
    def __init__(self,
                 config,
                 is_cross_attention=False,
                 layer_idx=None,
                 prefix_len=None):
        super().__init__(config, is_cross_attention, layer_idx)

        if prefix_len is not None:
            self.bias = torch.cat([
                torch.ones(*self.bias.size()[:3],
                           prefix_len,
                           dtype=self.bias.dtype), self.bias
            ],
                                  dim=-1)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1)**0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device)

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, :query_length, :key_length].to(
                torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value,
                                    dtype=attn_weights.dtype).to(
                                        attn_weights.device)
            attn_weights = torch.where(causal_mask,
                                       attn_weights.to(attn_weights.dtype),
                                       mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class PLModel(nn.Module):
    def __init__(self, config, role):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model_type = config.model_type
        assert 'opt' in self.model_type or 'gpt' in self.model_type
        self.role = role
        self.only_use_hidden_loss = config.only_use_hidden_loss

        dtype = torch.float16 if config.use_fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_type,
                                                          torch_dtype=dtype)
        if 'opt' in self.model_type:
            raw_embed_positions = self.model.model.decoder.embed_positions
            new_embed_positions = MyOPTLearnedPositionalEmbedding(
                self.model.config.max_position_embeddings,
                self.model.config.hidden_size,
                dtype=dtype)
            new_embed_positions.weight = raw_embed_positions.weight
            self.model.model.decoder.embed_positions = new_embed_positions

        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        self.n_head = self.model.config.num_attention_heads
        self.n_embd = self.hidden_size // self.n_head
        self.p_drop = self.model.config.dropout \
            if 'opt' in self.model_type else self.model.config.attn_pdrop
        self.n_layer = config.get(f'num_{role}_layers')
        self.share_layer_param = config.get(f'share_{role}_layer_param')
        if self.share_layer_param:
            start_layer_id = config.get(f'{role}_start_layer_id')
            num_layer_per_cell = config.get(f'num_{role}_layers_per_cell')
            assert self.n_layer % num_layer_per_cell == 0
            num_cells = self.n_layer // num_layer_per_cell
            if 'opt' in self.model_type:
                self.model.model.decoder.layers = nn.ModuleList(
                    list(self.model.model.decoder.
                         layers[start_layer_id:start_layer_id +
                                num_layer_per_cell]) * num_cells)
            else:
                self.model.transformer.h = nn.ModuleList(
                    list(self.model.transformer.
                         h[start_layer_id:start_layer_id + num_layer_per_cell])
                    * num_cells)
        else:
            if 'opt' in self.model_type:
                self.model.model.decoder.layers = \
                    self.model.model.decoder.layers[:self.n_layer]
            else:
                self.model.transformer.h = \
                    self.model.transformer.h[:self.n_layer]

        self.prefix_len = config.get(f'{role}_prefix_len')
        self.use_prefix_prj = config.use_prefix_prj
        if self.prefix_len > 0:
            if 'gpt' in self.model_type:
                for i in range(len(self.model.transformer.h)):
                    raw_attn = self.model.transformer.h[i].attn
                    new_attn = MyGPT2Attention(
                        self.model.config,
                        layer_idx=i,
                        prefix_len=self.prefix_len).to(dtype)
                    new_attn.c_attn = raw_attn.c_attn
                    new_attn.c_proj = raw_attn.c_proj
                    if hasattr(new_attn, 'q_attn'):
                        new_attn.q_attn = raw_attn.q_attn
                    self.model.transformer.h[i].attn = new_attn

            self.prefix_tokens = torch.arange(self.prefix_len).long()
            if self.use_prefix_prj:
                self.prefix_encoder = nn.Embedding(self.prefix_len,
                                                   self.hidden_size,
                                                   dtype=dtype)
                self.prefix_prj = nn.Sequential(
                    nn.Linear(self.hidden_size,
                              config.prefix_hidden_size,
                              dtype=dtype), nn.Tanh(),
                    nn.Linear(config.prefix_hidden_size,
                              self.n_layer * 2 * self.hidden_size,
                              dtype=dtype))
            else:
                self.prefix_encoder = nn.Embedding(self.prefix_len,
                                                   self.n_layer * 2 *
                                                   self.hidden_size,
                                                   dtype=dtype)

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

        if 'opt' in self.model_type:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                output_hidden_states=True,
            )
        else:
            position_ids = torch.arange(input_ids.size(1),
                                        dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(
                -1, input_ids.size(1))
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                output_hidden_states=True,
            )

        logits = outputs.logits
        loss = outputs.loss
        regular_loss, kd_loss = outputs.loss, None
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
        if self.use_prefix_prj:
            past_key_values = self.prefix_prj(past_key_values)
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd,
        )
        past_key_values = nn.functional.dropout(past_key_values,
                                                p=self.p_drop,
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
            return out

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False,
            batch_size=self.batch_size,
        )
