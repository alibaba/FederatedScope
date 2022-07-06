from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def get_rnn(model_config, local_data):
    from federatedscope.nlp.model.rnn import LSTM
    if isinstance(local_data, dict):
        if 'data' in local_data.keys():
            data = local_data['data']
        elif 'train' in local_data.keys():
            # local_data['train'] is Dataloader
            data = next(iter(local_data['train']))
        else:
            raise TypeError('Unsupported data type.')
    else:
        data = local_data

    x, _ = data

    # check the task
    if model_config.type == 'lstm':
        model = LSTM(in_channels=x.shape[1] if not model_config.in_channels
                     else model_config.in_channels,
                     hidden=model_config.hidden,
                     out_channels=model_config.out_channels,
                     embed_size=model_config.embed_size,
                     dropout=model_config.dropout)
    else:
        raise ValueError(f'No model named {model_config.type}!')

    return model


def get_transformer(model_config, local_data):
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
