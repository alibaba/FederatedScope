import logging

import federatedscope.register as register
from federatedscope.core.data.wrap_dataset import WrapDataset

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.model import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.model`, some modules are not '
        f'available.')


def get_shape_from_data(data, model_config, backend='torch'):
    """
    Extract the input shape from the given data, which can be used to build
    the data. Users can also use `data.input_shape` to specify the shape
    Arguments:
        data (`ClientData`): the data used for local training or evaluation
        The expected data format:
        1): {train/val/test: {x:ndarray, y:ndarray}}}
        2): {train/val/test: DataLoader}
    Returns:
        shape (tuple): the input shape
    """
    # Handle some special cases
    if model_config.type.lower() in ['vmfnet', 'hmfnet']:
        return data['train'].n_col if model_config.type.lower(
        ) == 'vmfnet' else data['train'].n_row
    elif model_config.type.lower() in [
            'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn'
    ] or model_config.type.startswith('gnn_'):
        num_label = data['num_label'] if 'num_label' in data else None
        num_edge_features = data['data'][
            'num_edge_features'] if model_config.type == 'mpnn' else None
        if model_config.task.startswith('graph'):
            # graph-level task
            data_representative = next(iter(data['train']))
            return data_representative.x.shape, num_label, num_edge_features
        else:
            # node/link-level task
            return data['data'].x.shape, num_label, num_edge_features

    if isinstance(data, dict):
        keys = list(data.keys())
        if 'test' in keys:
            key_representative = 'test'
        elif 'val' in keys:
            key_representative = 'val'
        elif 'train' in keys:
            key_representative = 'train'
        elif 'data' in keys:
            key_representative = 'data'
        else:
            key_representative = keys[0]
            logger.warning(f'We chose the key {key_representative} as the '
                           f'representative key to extract data shape.')
        data_representative = data[key_representative]
    else:
        # Handle the data with non-dict format
        data_representative = data

    if isinstance(data_representative, dict):
        if 'x' in data_representative:
            shape = data_representative['x'].shape
            if len(shape) == 1:  # (batch, ) = (batch, 1)
                return 1
            else:
                return shape
    elif backend == 'torch':
        import torch
        if issubclass(type(data_representative), torch.utils.data.DataLoader):
            x, _ = next(iter(data_representative))
            return x.shape
        else:
            try:
                x, _ = data_representative
                return x.shape
            except:
                raise TypeError('Unsupported data type.')
    elif backend == 'tensorflow':
        # TODO: Handle more tensorflow type here
        shape = data_representative['x'].shape
        if len(shape) == 1:  # (batch, ) = (batch, 1)
            return 1
        else:
            return shape


def get_model(model_config, local_data=None, backend='torch'):
    """
    Arguments:
        local_data (object): the model to be instantiated is
        responsible for the given data.
    Returns:
        model (torch.Module): the instantiated model.
    """
    if local_data is not None:
        input_shape = get_shape_from_data(local_data, model_config, backend)
    else:
        input_shape = model_config.input_shape

    if input_shape is None:
        logger.warning('The input shape is None. Please specify the '
                       '`data.input_shape`(a tuple) or give the '
                       'representative data to `get_model` if necessary')

    for func in register.model_dict.values():
        model = func(model_config, input_shape)
        if model is not None:
            return model

    if model_config.type.lower() == 'lr':
        if backend == 'torch':
            from federatedscope.core.lr import LogisticRegression
            model = LogisticRegression(in_channels=input_shape[-1],
                                       class_num=model_config.out_channels)
        elif backend == 'tensorflow':
            from federatedscope.cross_backends import LogisticRegression
            model = LogisticRegression(in_channels=input_shape[-1],
                                       class_num=1,
                                       use_bias=model_config.use_bias)
        else:
            raise ValueError

    elif model_config.type.lower() == 'mlp':
        from federatedscope.core.mlp import MLP
        model = MLP(channel_list=[input_shape[-1]] + [model_config.hidden] *
                    (model_config.layer - 1) + [model_config.out_channels],
                    dropout=model_config.dropout)

    elif model_config.type.lower() == 'quadratic':
        from federatedscope.tabular.model import QuadraticModel
        model = QuadraticModel(input_shape[-1], 1)

    elif model_config.type.lower() in ['convnet2', 'convnet5', 'vgg11', 'lr']:
        from federatedscope.cv.model import get_cnn
        model = get_cnn(model_config, input_shape)
    elif model_config.type.lower() in ['lstm']:
        from federatedscope.nlp.model import get_rnn
        model = get_rnn(model_config, input_shape)
    elif model_config.type.lower().endswith('transformers'):
        from federatedscope.nlp.model import get_transformer
        model = get_transformer(model_config, input_shape)
    elif model_config.type.lower() in [
            'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn'
    ]:
        from federatedscope.gfl.model import get_gnn
        model = get_gnn(model_config, input_shape)
    elif model_config.type.lower() in ['vmfnet', 'hmfnet']:
        from federatedscope.mf.model.model_builder import get_mfnet
        model = get_mfnet(model_config, input_shape)
    else:
        raise ValueError('Model {} is not provided'.format(model_config.type))

    return model


def get_trainable_para_names(model):
    return set(dict(list(model.named_parameters())).keys())
