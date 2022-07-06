from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.cv.model.cnn import ConvNet2, ConvNet5, VGG11


def get_cnn(model_config, local_data):
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
    if model_config.type == 'convnet2':
        model = ConvNet2(in_channels=x.shape[1],
                         h=x.shape[2],
                         w=x.shape[3],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_config.type == 'convnet5':
        model = ConvNet5(in_channels=x.shape[1],
                         h=x.shape[2],
                         w=x.shape[3],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_config.type == 'vgg11':
        model = VGG11(in_channels=x.shape[1],
                      h=x.shape[2],
                      w=x.shape[3],
                      hidden=model_config.hidden,
                      class_num=model_config.out_channels,
                      dropout=model_config.dropout)
    else:
        raise ValueError(f'No model named {model_config.type}!')

    return model
