import gc
import copy
import logging
import torch
import torch.nn as nn

from federatedscope.llm.model.adapter_builder import AdapterModel

logger = logging.getLogger(__name__)


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
    # TODO: support more LLM
    layers = model.model.transformer.h

    emulator = COMP_FUNC_MAPPING[strategy](layers[l:r], **kwargs)

    for param in emulator.parameters():
        param.data = param.data.float()
        param.requires_grad = False

    emulator_adapter = nn.ModuleList()
    for idx in range(l):
        emulator_adapter.append(layers[idx])

    for idx in range(len(emulator)):
        emulator_adapter.append(emulator[idx])

    for idx in range(r, len(layers)):
        emulator_adapter.append(layers[idx])

    new_model = copy.deepcopy(model)

    # TODO: support more LLM
    new_model.model.transformer.h = emulator_adapter

    gc.collect()
    torch.cuda.empty_cache()

    return new_model
