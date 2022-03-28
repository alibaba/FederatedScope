from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.nlp.model.rnn import LSTM


def get_rnn(model_config, local_data):
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
                     embed_size=model_config.embed_size)
    else:
        raise ValueError(f'No model named {model_config.type}!')

    return model
