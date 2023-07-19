import gc
import os
import copy
import logging
import torch
import torch.nn as nn

from transformers import (OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM,
                          LlamaForCausalLM)
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.offsite_tuning.kd_trainer import KDTrainer
from federatedscope.core.auxiliaries.data_builder import get_data

logger = logging.getLogger(__name__)


def add_prologue(module, prologue):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
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


def add_epilogue(module, epilogue):
    """
    This function is borrowed from offsite-tuning:
    https://github.com/mit-han-lab/offsite-tuning/blob/main/offsite_tuning
    /utils.py
    """
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
    elif isinstance(adapter_model.model, LlamaForCausalLM):
        layers = adapter_model.model.model.layers
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
    elif isinstance(adapter_model.model, LlamaForCausalLM):
        adapter_model.model.model.layers = layers
    else:
        # TODO: support more LLM
        logger.warning(f'Model {type(adapter_model.model)} not support, '
                       f'use default setting.')
        adapter_model.model.transformer.h = layers
    adapter_model.student = layers
    add_prologue(adapter_model.student[0], None)
    add_epilogue(adapter_model.student[-1], None)
    adapter_model.student_l = adapter_model.student[0]
    adapter_model.student_r = adapter_model.student[-1]
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
                                  emulator_r=1000,
                                  **kwargs):
    layers = get_layers(model)
    l, r = max(emulator_l, 1), min(emulator_r, len(layers) - 1)

    # Set the to-compress part untrainable
    for layer in layers[l:r]:
        for param in layer.parameters():
            param.data = param.data.float()
            param.requires_grad = False
    # Set teacher model
    model.teacher = layers[l:r]

    emulator = COMP_FUNC_MAPPING[strategy](layers[l:r], **kwargs)

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
    # Set student model
    new_model = set_layers(new_model, emulator_and_adapter)

    gc.collect()
    torch.cuda.empty_cache()

    return new_model


def align_student_with_teacher(raw_model, adap_model, cfg, device, monitor):
    def build_cfg_for_alignment(config):
        new_cfg = copy.deepcopy(config)
        new_cfg.defrost()

        # Overwrite `config.train` with
        # `config.llm.offsite_tuning.emu_align.train`
        for key, value in \
                new_cfg.llm.offsite_tuning.emu_align.train.optimizer.items():
            if key.startswith('__'):
                continue
            setattr(new_cfg, f'train.optimizer.{key}', value)
        new_cfg.train.local_update_steps = \
            config.llm.offsite_tuning.emu_align.train.local_update_steps
        new_cfg.train.batch_or_epoch = \
            config.llm.offsite_tuning.emu_align.train.batch_or_epoch

        # Overwrite `config.data` with
        # `config.llm.offsite_tuning.emu_align.data`
        for key, value in \
                new_cfg.llm.offsite_tuning.emu_align.data.items():
            if key.startswith('__'):
                continue
            setattr(new_cfg, f'data.{key}', value)
        # Used for data translator
        new_cfg.federate.client_num = 1

        # TODO: might generate extra cfg file, delete
        new_cfg.freeze()
        return new_cfg

    does_train_emulator = True
    if cfg.llm.offsite_tuning.emu_align.restore_from != '':
        try:
            if not os.path.exists(
                    cfg.llm.offsite_tuning.emu_align.restore_from):
                logger.warning(
                    f'Invalid `emu_align.restore_from`:'
                    f' {cfg.llm.offsite_tuning.emu_align.restore_from}.')
            else:
                assert adap_model is not None
                ckpt = torch.load(
                    cfg.llm.offsite_tuning.emu_align.restore_from,
                    map_location='cpu')
                adap_model.load_state_dict(ckpt['model'], strict=False)
                logger.info("Restored the model from ckpt")
                logger.warning(
                    "Please make sure the dtype of model keep the same.")
                # Make student un-trainable
                for layer in adap_model.student:
                    for param in layer.parameters():
                        param.requires_grad = False
                does_train_emulator = False
        except Exception as error:
            logger.error(error)

    if not does_train_emulator:
        return adap_model

    new_cfg = build_cfg_for_alignment(cfg)

    # Make student trainable
    for layer in adap_model.student:
        for param in layer.parameters():
            param.requires_grad = True

    # Loading held-out data
    logger.info('Loading held-out dataset for alignment...')
    data, modified_cfg = get_data(new_cfg.clone())
    new_cfg.merge_from_other_cfg(modified_cfg)

    # Create `KDTrainer` and train
    kd_trainer = KDTrainer(raw_model,
                           adap_model,
                           data[1],
                           device,
                           new_cfg,
                           only_for_eval=False,
                           monitor=monitor)
    logger.info('Start to align student model with teacher model...')
    kd_trainer.train()
    logger.info('Alignment finished!')

    # Save aligned model
    adap_model.save_model(cfg.llm.offsite_tuning.emu_align.save_to)

    # Make student un-trainable
    for layer in adap_model.student:
        for param in layer.parameters():
            param.requires_grad = False

    return adap_model
