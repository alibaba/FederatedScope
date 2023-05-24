import gc
import logging
import torch
import torch.nn as nn
# TODO: Use `federatedscope.llm.model.AdapterModule`
from torch.nn import Module as AdapterModule

from transformers import (GPT2LMHeadModel, LlamaForCausalLM)

logger = logging.getLogger(__name__)


def add_prologue(module, prologue=None):
    module.old_forward = module.forward
    module.prologue = prologue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            self.input_args = args
            self.input_kwargs = kwargs
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            args = (x, ) + args[1:]
            return self.old_forward(*args, **kwargs)

        return lambda_forward

    module.forward = new_forward(module)
    return module


def add_epilogue(module, epilogue=None):
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)

            if isinstance(output, tuple):
                output = (x, ) + output[1:]
            else:
                output = x

            self.cached_output = x
            return output

        return lambda_forward

    module.forward = new_forward(module)
    return module


def model_drop_layer(model, drop_ratio=0.5, **kwargs):
    def get_llm_layers(model):
        if isinstance(model, GPT2LMHeadModel):
            layers = model.transformer.h
        elif isinstance(model, LlamaForCausalLM):
            layers = model.layers.h
        else:
            logger.warning(f'Model drop layer might cause '
                           f'error with unknown LLM {model}')
            layers = model.transformer.h
        return layers

    layers = get_llm_layers(model)
    new_model = nn.ModuleList()

    num_new_layers = round(len(layers) * (1 - drop_ratio))

    stride = (len(layers) - 1) * (num_new_layers - 1)

    for i in range(num_new_layers):
        idx = round(i * stride)
        logger.info(f"Adding layer {idx} to student")
        new_model.append(layers[idx])

    raise new_model


def model_pruning(model, ratio=0.5, **kwargs):
    raise NotImplementedError


def model_quantization(model, bits, **kwargs):
    raise NotImplementedError


def model_distillation(model, **kwargs):
    raise NotImplementedError


def merge_emulator_and_adapter(emulator, adapter):
    # TODO: Implement this
    # model.emulator = emulator
    # model.adapter = adapter
    # model.forward = ...
    ...


def generate_emulator_and_adapter(model: AdapterModule,
                                  strategy='drop_layer',
                                  **kwargs):
    # TODO: Delete this line
    return model

    COMP_FUNC_MAP = {
        'drop_layer': model_drop_layer,
        'pruning': model_pruning,
        'quantization': model_quantization,
        'distillation': model_distillation
    }
    # TODO: update with `federatedscope.llm.model.AdapterModule` enabled
    # base_model = model.base_model
    base_model = model
    emulator = COMP_FUNC_MAP[strategy](base_model, **kwargs)

    for param in emulator.parameters():
        param.data = param.data.float()
        param.requires_grad = False

    # TODO: Implement this
    new_model = merge_emulator_and_adapter(emulator, model.adapter)

    gc.collect()
    torch.cuda.empty_cache()

    return new_model
