import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.model import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.model`, some modules are not '
        f'available.')


def get_model(model_config, local_data, backend='torch'):
    """
    Arguments:
        local_data (object): the model to be instantiated is
        responsible for the given data.
    Returns:
        model (torch.Module): the instantiated model.
    """
    for func in register.model_dict.values():
        model = func(model_config, local_data)
        if model is not None:
            return model

    if model_config.type.lower() == 'lr':
        if backend == 'torch':
            from federatedscope.core.lr import LogisticRegression
            # TODO: make the instantiation more general
            if isinstance(
                    local_data, dict
            ) and 'test' in local_data and 'x' in local_data['test']:
                model = LogisticRegression(
                    in_channels=local_data['test']['x'].shape[-1],
                    class_num=1,
                    use_bias=model_config.use_bias)
            else:
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
                model = LogisticRegression(in_channels=x.shape[-1],
                                           class_num=model_config.out_channels)
        elif backend == 'tensorflow':
            from federatedscope.cross_backends import LogisticRegression
            model = LogisticRegression(
                in_channels=local_data['test']['x'].shape[-1],
                class_num=1,
                use_bias=model_config.use_bias)
        else:
            raise ValueError

    elif model_config.type.lower() == 'mlp':
        from federatedscope.core.mlp import MLP
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
        model = MLP(channel_list=[x.shape[-1]] + [model_config.hidden] *
                    (model_config.layer - 1) + [model_config.out_channels],
                    dropout=model_config.dropout)

    elif model_config.type.lower() == 'quadratic':
        from federatedscope.tabular.model import QuadraticModel
        if isinstance(local_data, dict):
            data = next(iter(local_data['train']))
        else:
            # TODO: complete the branch
            data = local_data
        x, _ = data
        model = QuadraticModel(x.shape[-1], 1)

    elif model_config.type.lower() in ['convnet2', 'convnet5', 'vgg11', 'lr']:
        from federatedscope.cv.model import get_cnn
        model = get_cnn(model_config, local_data)
    elif model_config.type.lower() in ['lstm']:
        from federatedscope.nlp.model import get_rnn
        model = get_rnn(model_config, local_data)
    elif model_config.type.lower().endswith('transformers'):
        from federatedscope.nlp.model import get_transformer
        model = get_transformer(model_config, local_data)
    elif model_config.type.lower() in [
            'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn'
    ]:
        from federatedscope.gfl.model import get_gnn
        model = get_gnn(model_config, local_data)
    elif model_config.type.lower() in ['vmfnet', 'hmfnet']:
        from federatedscope.mf.model.model_builder import get_mfnet
        model = get_mfnet(model_config, local_data)
    else:
        raise ValueError('Model {} is not provided'.format(model_config.type))

    return model


def get_trainable_para_names(model):
    return set(dict(list(model.named_parameters())).keys())
