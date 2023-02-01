import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from federatedscope.nlp.hetero_tasks.dataset.utils import setup_tokenizer
from federatedscope.nlp.loss.label_smooth_loss import \
    LabelSmoothingLoss


class ModelOutput(object):
    def __init__(self,
                 loss=None,
                 regular_loss=None,
                 contrastive_loss=None,
                 logits=None,
                 hidden_states=None,
                 example_indices=None):
        self.loss = loss
        self.regular_loss = regular_loss
        self.contrastive_loss = contrastive_loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.example_indices = example_indices


class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, inner_dim, out_dim, dropout_prob):
        super().__init__()

        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_prj = nn.Linear(inner_dim, out_dim)

    def forward(self, x):
        x = self.dense(self.dropout(x))
        x = torch.tanh(x)
        x = self.out_prj(self.dropout(x))
        return x


class ATCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.encoder_decoder import EncoderDecoderModel
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            config.model_type, config.model_type)
        self.lm_head = BertLMPredictionHead(self.model.encoder.config)

        self.client_id = None
        self.task = config.task
        self.pt_cfg = self.model.encoder.config
        self.vocab_size = self.pt_cfg.vocab_size
        self.hidden_size = self.pt_cfg.hidden_size
        self.dropout_prob = self.pt_cfg.hidden_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        setup_tokenizer(config.model_type)  # update global token ids
        from federatedscope.nlp.hetero_tasks.dataset.utils import \
            BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
        self.label_smoothing = config.label_smoothing
        self.padding_idx = PAD_TOKEN_ID
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        self.use_contrastive_loss = config.use_contrastive_loss
        if self.use_contrastive_loss:
            self.contrast_topk = config.contrast_topk
            self.contrast_temp = config.contrast_temp
            self.contrast_head = ContrastiveHead(
                input_dim=self.hidden_size,
                inner_dim=self.hidden_size,
                out_dim=self.hidden_size,
                dropout_prob=self.dropout_prob)

        # for eval generation
        self.model.config.decoder_start_token_id = BOS_TOKEN_ID
        self.model.config.eos_token_id = EOS_TOKEN_ID
        self.model.config.pad_token_id = PAD_TOKEN_ID
        self.model.config.vocab_size = self.pt_cfg.vocab_size
        self.model.config.max_length = config.max_length
        self.model.config.min_length = config.min_length
        self.model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.model.config.length_penalty = config.length_penalty
        self.model.config.num_beams = config.num_beams

    def update_client_id(self, client_id):
        self.client_id = client_id

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
        labels=None,
        pretrain_task=None,
        contrast_monitor=None,
        in_contrast_prepare=None,
        example_indices=None,
    ):
        if in_contrast_prepare:  # return dec_hidden_states & dec_out
            self.eval()
            with torch.no_grad():
                example_indices = [
                    k for k in example_indices
                    if k.item() in contrast_monitor.synth_tokens
                ]
                if len(example_indices) == 0:
                    return ModelOutput(example_indices=example_indices)

                example_indices = torch.stack(example_indices)
                synth_input_ids = torch.stack([
                    contrast_monitor.synth_tokens[k.item()]
                    for k in example_indices
                ]).to(self.model.device)

                enc_hidden = torch.stack([
                    contrast_monitor.enc_hidden[k.item()]
                    for k in example_indices
                ]).to(self.model.device)
                outputs = self.model.decoder.bert(
                    input_ids=synth_input_ids,
                    encoder_hidden_states=enc_hidden,
                )
                logits = self.model.decoder.cls(outputs.last_hidden_state)
                dec_hidden = self.contrast_head(
                    outputs.last_hidden_state).mean(1)

                return ModelOutput(logits=logits.argmax(-1),
                                   hidden_states=dec_hidden,
                                   example_indices=example_indices)

        enc_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        regular_loss, contrastive_loss = None, None
        if self.task == 'pretrain':
            if pretrain_task == 'mlm':
                logits = self.lm_head(enc_outputs.last_hidden_state)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(logits.view(-1, self.vocab_size),
                                          labels.view(-1))
                loss = masked_lm_loss

            elif pretrain_task == 'denoise':
                dec_outputs = self.model.decoder.bert(
                    input_ids=labels,
                    encoder_hidden_states=enc_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                logits = self.model.decoder.cls(
                    dec_outputs.last_hidden_state)[:, :-1, :]
                loss_fct = CrossEntropyLoss(ignore_index=self.padding_idx)
                denoise_loss = loss_fct(
                    logits.contiguous().view(-1, self.vocab_size),
                    labels[:, 1:].contiguous().view(-1))
                loss = denoise_loss

            else:
                raise KeyError(
                    'Unsupported pretrain task: \'{}\''.format(pretrain_task))

        else:
            # regular loss
            if self.task in {'imdb', 'agnews'}:
                pooled_output = self.dropout(enc_outputs.pooler_output)
                logits = self.classifier(pooled_output)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)),
                                labels.view(-1))

            elif self.task in {'squad', 'newsqa'}:
                logits = self.classifier(enc_outputs.last_hidden_state)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1).contiguous()
                end_logits = end_logits.squeeze(-1).contiguous()
                logits = (start_logits, end_logits)

                # sometimes the start/end positions are outside our model
                # inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

            elif self.task in {'cnndm', 'msqg'}:
                dec_outputs = self.model.decoder.bert(
                    input_ids=labels,
                    encoder_hidden_states=enc_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                dec_hidden_states = dec_outputs.last_hidden_state
                logits = self.model.decoder.cls(dec_hidden_states)[:, :-1, :]

                num_tokens = labels[:, 1:].ne(self.padding_idx).sum().item()
                label_smoothing = self.label_smoothing if self.training \
                    else 0.0
                if label_smoothing > 0:
                    loss_fct = LabelSmoothingLoss(
                        label_smoothing,
                        self.vocab_size,
                        ignore_index=self.padding_idx,
                    ).to(logits.device)
                    loss = loss_fct(
                        F.log_softmax(logits.contiguous().view(
                            -1, self.vocab_size),
                                      dim=-1),
                        labels[:, 1:].contiguous().view(-1)) / num_tokens
                else:
                    loss_fct = CrossEntropyLoss(ignore_index=self.padding_idx)
                    loss = loss_fct(
                        logits.contiguous().view(-1, self.vocab_size),
                        labels[:, 1:].contiguous().view(-1))

            else:
                raise KeyError('Unsupported task: \'{}\''.format(self.task))

            # contrastive loss
            if self.use_contrastive_loss and self.training:
                regular_loss = loss.clone()
                example_indices = [
                    k for k in example_indices
                    if k.item() in contrast_monitor.synth_tokens
                ]
                all_group_ids = contrast_monitor.all_group_ids[self.client_id]
                topk_group_ids = \
                    contrast_monitor.topk_group_ids[self.client_id]
                if len(example_indices) > 0 and len(topk_group_ids) > 1:
                    example_indices = torch.stack(example_indices)
                    synth_input_ids = torch.stack([
                        contrast_monitor.synth_tokens[k.item()]
                        for k in example_indices
                    ]).to(self.model.device)

                    contrast_enc_hidden = torch.stack([
                        contrast_monitor.enc_hidden[k.item()]
                        for k in example_indices
                    ]).to(self.model.device)
                    contrast_outputs = self.model.decoder.bert(
                        input_ids=synth_input_ids,
                        encoder_hidden_states=contrast_enc_hidden,
                    )
                    cur_dec_hidden = self.contrast_head(
                        contrast_outputs.last_hidden_state).mean(1)

                    pos_client_ids = [
                        x for x in topk_group_ids[1:self.contrast_topk + 1]
                    ]
                    all_dec_hiddens = contrast_monitor.dec_hidden
                    sim_hiddens = [[
                        all_dec_hiddens[cid][k.item()] for k in example_indices
                    ] for cid in pos_client_ids]
                    sim_hiddens = torch.stack([
                        torch.stack(hid) for hid in sim_hiddens
                    ]).mean(0).to(self.model.device)
                    sim_matrix = F.cosine_similarity(cur_dec_hidden,
                                                     sim_hiddens,
                                                     dim=-1)
                    nominator = torch.exp(sim_matrix / self.contrast_temp)
                    denominator = nominator

                    neg_client_ids = [
                        x for x in all_group_ids[::-1][:self.contrast_topk]
                        if x not in topk_group_ids
                    ]
                    if len(neg_client_ids) > 0:
                        dissim_hiddens = [[
                            all_dec_hiddens[cid][k.item()]
                            for k in example_indices
                        ] for cid in neg_client_ids]
                        dissim_hiddens = torch.stack([
                            torch.stack(hid) for hid in dissim_hiddens
                        ]).to(self.model.device)
                        dissim_matrix = F.cosine_similarity(
                            cur_dec_hidden.unsqueeze(0),
                            dissim_hiddens,
                            dim=-1)
                        denominator = denominator + (torch.exp(
                            dissim_matrix / self.contrast_temp)).sum(0)

                    contrastive_loss = -torch.log(
                        nominator / denominator).mean()
                    loss += contrastive_loss

        return ModelOutput(loss=loss,
                           regular_loss=regular_loss,
                           contrastive_loss=contrastive_loss,
                           logits=logits)
