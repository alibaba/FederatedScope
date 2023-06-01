import gc
import copy
import logging
import torch
import torch.nn as nn

from transformers import (
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
)
from federatedscope.llm.model.adapter_builder import AdapterModel

logger = logging.getLogger(__name__)


def get_layers(adapter_model):
    """
    Modified from the official implementation:
    https://github.com/mit-han-lab/offsite-tuning/tree/main
    """
    if isinstance(adapter_model.model, OPTForCausalLM):
        layers = adapter_model.model.model.decoder.layers
    elif isinstance(adapter_model.model, GPT2LMHeadModel):
        layers = adapter_model.model.transformer.h
    elif isinstance(adapter_model.model, BloomForCausalLM):
        layers = adapter_model.model.transformer.h
    else:
        # TODO: support more LLM
        logger.warning(f'Model {type(adapter_model.model)} not support, '
                       f'use default setting.')
        layers = adapter_model.model.transformer.h
    return layers


def set_layers(adapter_model, layers):
    if isinstance(adapter_model.model, OPTForCausalLM):
        adapter_model.model.model.decoder.layers = layers
    elif isinstance(adapter_model.model, GPT2LMHeadModel):
        adapter_model.model.transformer.h = layers
    elif isinstance(adapter_model.model, BloomForCausalLM):
        adapter_model.model.transformer.h = layers
    else:
        # TODO: support more LLM
        logger.warning(f'Model {type(adapter_model.model)} not support, '
                       f'use default setting.')
        adapter_model.model.transformer.h = layers
    return adapter_model


def model_drop_layer(layers, drop_ratio=0.5, **kwargs):
    new_model = nn.ModuleList()
    num_new_layers = round(len(layers) * (1 - drop_ratio))

    stride = (len(layers) - 1) / (num_new_layers - 1)

    for i in range(num_new_layers):
        idx = int(i * stride)
        logger.info(f"Adding layer {idx} to emulator.")
        new_model.append(layers[idx])

    return new_model


def model_pruning(model, ratio=0.5, **kwargs):
    raise NotImplementedError


def model_quantization(model, bits, **kwargs):
    raise NotImplementedError


def model_distillation(model, **kwargs):
    raise NotImplementedError


COMP_FUNC_MAPPING = {
    'drop_layer': model_drop_layer,
    'pruning': model_pruning,
    'quantization': model_quantization,
    'distillation': model_distillation
}


def generate_emulator_and_adapter(model: AdapterModel,
                                  strategy='drop_layer',
                                  emulator_l=1,
                                  emulator_r=10,
                                  **kwargs):
    l, r = emulator_l, emulator_r
    layers = get_layers(model)

    emulator = COMP_FUNC_MAPPING[strategy](layers[l:r], **kwargs)

    for param in emulator.parameters():
        param.data = param.data.float()
        param.requires_grad = False

    emulator_and_adapter = nn.ModuleList()

    # Adapter before Emulator
    for idx in range(l):
        emulator_and_adapter.append(layers[idx])

    # Emulator
    for idx in range(len(emulator)):
        emulator_and_adapter.append(emulator[idx])

    # Adapter after Emulator
    for idx in range(r, len(layers)):
        emulator_and_adapter.append(layers[idx])

    new_model = copy.deepcopy(model)
    new_model = set_layers(new_model, emulator_and_adapter)

    gc.collect()
    torch.cuda.empty_cache()

    return new_model
