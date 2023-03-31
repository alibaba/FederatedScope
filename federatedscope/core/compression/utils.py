import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _symmetric_uniform_quantization(x, nbits, stochastic=False):
    assert (torch.isnan(x).sum() == 0)
    assert (torch.isinf(x).sum() == 0)

    c = torch.max(torch.abs(x))
    s = c / (2**(nbits - 1) - 1)
    if s == 0:
        return x, s
    c_minus = c * -1.0

    # qx = torch.where(x.ge(c), c, x)
    # qx = torch.where(qx.le(c_minus), c_minus, qx)
    # qx.div_(s)
    qx = x / s

    if stochastic:
        noise = qx.new(qx.shape).uniform_(-0.5, 0.5)
        qx.add_(noise)

    qx.clamp_(-(2**(nbits - 1) - 1), (2**(nbits - 1) - 1)).round_()
    return qx, s


def symmetric_uniform_quantization(state_dict, nbits=8):
    """
    Perform symmetric uniform quantization to weight in conv & fc layers
    Args:
        state_dict: dict of model parameter (torch_model.state_dict)
        nbits: the bit of values after quantized, chosen from [8, 16]

    Returns:
        The quantized model parameters
    """
    if nbits == 8:
        quant_data_type = torch.int8
    elif nbits == 16:
        quant_data_type = torch.int16
    else:
        logger.info(f'The provided value of nbits ({nbits}) is invalid, and we'
                    f' change it to 8')
        nbits = 8
        quant_data_type = torch.int8

    quant_state_dict = dict()
    for key, value in state_dict.items():
        if ('fc' in key or 'conv' in key) and 'weight' == key.split('.')[-1]:
            q_weight, w_s = _symmetric_uniform_quantization(value, nbits=nbits)
            quant_state_dict[key.replace(
                'weight', 'weight_quant')] = q_weight.type(quant_data_type)
            quant_state_dict[key.replace('weight', 'weight_scale')] = w_s
        else:
            quant_state_dict[key] = value

    return quant_state_dict


def symmetric_uniform_dequantization(state_dict):
    """
    Perform symmetric uniform dequantization
    Args:
        state_dict: dict of model parameter (torch_model.state_dict)

    Returns:
        The model parameters after dequantization
    """
    dequantizated_state_dict = dict()
    for key, value in state_dict.items():
        if 'weight_quant' in key:
            alpha = state_dict[key.replace('weight_quant', 'weight_scale')]
            dequantizated_state_dict[key.replace('weight_quant',
                                                 'weight')] = value * alpha
        elif 'weight_scale' in key:
            pass
        else:
            dequantizated_state_dict[key] = value

    return dequantizated_state_dict
