from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.models.bert import BertModel


class ModelOutput(object):
    def __init__(self, loss, logits, hidden_states, attentions):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


# Build your torch or tf model class here
class MyModel(nn.Module):
    def __init__(self, config):  # server config
        super().__init__()

        self.bert = BertModel.from_pretrained(config.bert_type)

        if config.maml:
            for n, p in self.bert.named_parameters():
                if not (n.startswith('pooler') or n.startswith('encoder.layer.11.output')):
                    p.requires_grad = False
                else:
                    print(n)

        self.hidden_size = self.bert.config.hidden_size
        self.dropout_prob = self.bert.config.hidden_dropout_prob
        self.config = config

        # For NLU
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.ModuleDict()
        all_tasks = [k for k in config.num_labels.keys() if k != 'cfg_check_funcs']
        self.all_labels = {k: config.num_labels[k] for k in all_tasks}
        for t, num_lb in self.all_labels.items():
            num_lb = config.num_labels[t]
            if num_lb is not None:
                self.classifier[t] = nn.Linear(self.hidden_size, num_lb)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        config=None,  # client specific config
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        task = config.data.type
        num_labels = self.all_labels[task]

        if task == 'sts':
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier[task](pooled_output)

            loss = None
            if labels is not None:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels.view(-1))

        elif task == 'imdb':
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier[task](pooled_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        elif task == 'squad':
            sequence_output = outputs.last_hidden_state
            logits = self.classifier[task](sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            logits = (start_logits, end_logits)

            loss = None
            if start_positions is not None and end_positions is not None:
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    model = MyModel(model_config)
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "text-dt":
        model = ModelBuilder(model_config, local_data)
        return model
