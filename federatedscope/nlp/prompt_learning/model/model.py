import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from federatedscope.register import register_model


class ModelOutput(object):
    def __init__(self, loss=None, logits=None, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states


class PLModel(nn.Module):
    def __init__(self, config, role='client'):
        super().__init__()
        from transformers import BertModel

        self.role = role
        self.num_labels = config.num_labels
        self.model = BertModel.from_pretrained(config.model_type)

        self.hidden_size = self.model.config.hidden_size
        self.n_layer = self.model.config.num_hidden_layers
        self.n_head = self.model.config.num_attention_heads
        self.n_embd = self.hidden_size // self.n_head

        if role == 'client' and 0 < config.num_client_layers < self.n_layer:
            self.model.encoder.layer = \
                self.model.encoder.layer[:config.num_client_layers]
            self.n_layer = config.num_client_layers

        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.prefix_len = config.prefix_len
        self.prefix_tokens = torch.arange(self.prefix_len).long()
        self.prefix_encoder = nn.Embedding(self.prefix_len,
                                           self.n_layer * 2 * self.hidden_size)

        self.freeze_param = config.freeze_param
        self._freeze_param()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        batch_size = input_ids.size(0)
        multi_choice = input_ids.ndim == 3
        if multi_choice:  # copa
            num_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            past_key_values = self._get_prompt(batch_size * num_choices)
            prefix_attention_mask = torch.ones(batch_size * num_choices,
                                               self.prefix_len).to(
                                                   attention_mask.device)
            prefix_attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=-1)
        else:
            past_key_values = self._get_prompt(batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(
                attention_mask.device)
            prefix_attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model(
            input_ids,
            attention_mask=prefix_attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
        )

        logits = self.classifier(self.dropout(outputs.pooler_output))
        if multi_choice:
            logits = logits.view(-1, num_choices)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def _get_prompt(self, batch_size):
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
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def _freeze_param(self):
        for n, p in self.named_parameters():
            for fp in self.freeze_param:
                if fp in n:
                    p.requires_grad = False
                    break
