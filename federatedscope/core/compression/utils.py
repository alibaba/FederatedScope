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
        nbits: the bit of values after quantized, chosen from [4, 8, 16]

    Returns:
        The quantized model parameters
    """
    if nbits == 4:
        quant_data_type = torch.uint8
    elif nbits == 8:
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
            if nbits == 4:
                flat = q_weight.flatten().to(torch.int8)
                orig_shape = list(value.shape)
                pad = 1 if flat.numel() % 2 != 0 else 0
                unsigned = (flat + 8).to(torch.uint8)
                if pad > 0:
                    unsigned = torch.nn.functional.pad(unsigned, (0, 1),
                                                       value=8)
                packed = ((unsigned[0::2] << 4) | (unsigned[1::2] & 0x0F)).to(
                    torch.uint8)
                quant_state_dict[key.replace('weight',
                                             'weight_quant')] = packed
                quant_state_dict[key.replace('weight', 'weight_scale')] = w_s
                quant_state_dict[key.replace('weight',
                                             'weight_shape')] = orig_shape
            else:
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
            shape_key = key.replace('weight_quant', 'weight_shape')
            if shape_key in state_dict:
                orig_shape = state_dict[shape_key]
                uq0 = (value >> 4) & 0x0F
                uq1 = value & 0x0F
                unpacked = torch.empty(value.numel() * 2,
                                       dtype=torch.int8,
                                       device=value.device)
                unpacked[0::2] = uq0.to(torch.int8) - 8
                unpacked[1::2] = uq1.to(torch.int8) - 8
                num_elements = 1
                for dim in orig_shape:
                    num_elements *= dim
                unpacked = unpacked[:num_elements].reshape(orig_shape)
                dequantizated_state_dict[key.replace(
                    'weight_quant', 'weight')] = unpacked.float() * alpha
            else:
                dequantizated_state_dict[key.replace('weight_quant',
                                                     'weight')] = value * alpha
        elif 'weight_scale' in key or 'weight_shape' in key:
            pass
        else:
            dequantizated_state_dict[key] = value

    return dequantizated_state_dict
