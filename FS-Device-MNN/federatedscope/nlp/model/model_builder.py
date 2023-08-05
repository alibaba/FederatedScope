from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def get_rnn(model_config, input_shape):
    from federatedscope.nlp.model.rnn import LSTM
    # check the task
    # input_shape: (batch_size, seq_len, hidden) or (seq_len, hidden)
    if model_config.type == 'lstm':
        model = LSTM(
            in_channels=input_shape[-2]
            if not model_config.in_channels else model_config.in_channels,
            hidden=model_config.hidden,
            out_channels=model_config.out_channels,
            embed_size=model_config.embed_size,
            dropout=model_config.dropout)
    else:
        raise ValueError(f'No model named {model_config.type}!')

    return model


def get_transformer(model_config, input_shape):
    from transformers import AutoModelForPreTraining, \
        AutoModelForQuestionAnswering, AutoModelForSequenceClassification, \
        AutoModelForTokenClassification, AutoModelWithLMHead, AutoModel

    model_func_dict = {
        'PreTraining'.lower(): AutoModelForPreTraining,
        'QuestionAnswering'.lower(): AutoModelForQuestionAnswering,
        'SequenceClassification'.lower(): AutoModelForSequenceClassification,
        'TokenClassification'.lower(): AutoModelForTokenClassification,
        'WithLMHead'.lower(): AutoModelWithLMHead,
        'Auto'.lower(): AutoModel
    }
    assert model_config.task.lower(
    ) in model_func_dict, f'model_config.task should be in' \
                          f' {model_func_dict.keys()} ' \
                          f'when using pre_trained transformer model '
    path, _ = model_config.type.split('@')
    model = model_func_dict[model_config.task.lower()].from_pretrained(
        path, num_labels=model_config.out_channels)

    return model
