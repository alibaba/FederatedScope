from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.cv.model.cnn import ConvNet2, ConvNet5, VGG11


def get_cnn(model_config, input_shape):
    # check the task
    # input_shape: (batch_size, in_channels, h, w) or (in_channels, h, w)
    if model_config.type == 'convnet2':
        model = ConvNet2(in_channels=input_shape[-3],
                         h=input_shape[-2],
                         w=input_shape[-1],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_config.type == 'convnet5':
        model = ConvNet5(in_channels=input_shape[-3],
                         h=input_shape[-2],
                         w=input_shape[-1],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_config.type == 'vgg11':
        model = VGG11(in_channels=input_shape[-3],
                      h=input_shape[-2],
                      w=input_shape[-1],
                      hidden=model_config.hidden,
                      class_num=model_config.out_channels,
                      dropout=model_config.dropout)
    else:
        raise ValueError(f'No model named {model_config.type}!')

    return model
