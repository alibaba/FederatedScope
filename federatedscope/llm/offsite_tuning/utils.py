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


def set_layers(adapter_model, layers, emu_l=0, emu_r=-1):
    """
    Set the layers of the adapter model based on the model type and the
    emulator range.

    Args:
        adapter_model (AdapterModel): The adapter model object that contains
            the causal language model and the adapter layers.
        layers (nn.ModuleList): The list of layers to be assigned to the
            adapter model.
        emu_l (int): The left index of the emulator range. Default to 0.
        emu_r (int): The right index of the emulator range. Default to -1.

    Returns:
        AdapterModel: The adapter model object with the updated layers.
    """
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
    adapter_model.student = layers[emu_l:emu_r]
    adapter_model.adapter = layers[:emu_l] + layers[emu_r:]
    add_prologue(adapter_model.student[0], None)
    add_epilogue(adapter_model.student[-1], None)
    adapter_model.student_l = adapter_model.student[0]
    adapter_model.student_r = adapter_model.student[-1]
    return adapter_model


def model_drop_layer(layers, drop_ratio=0.5, **kwargs):
    """
    Drop layers from a list of layers based on a drop ratio.

    Args:
        layers (nn.ModuleList): The list of layers to be dropped.
        drop_ratio (float): The ratio of layers to be dropped. Default to 0.5.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.ModuleList: A new list of layers with some layers dropped.
    """
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
    model.teacher = layers[l:r]  # Ref for old model

    emulator = COMP_FUNC_MAPPING[strategy](layers[l:r], **kwargs)

    emulator_and_adapter = nn.ModuleList()

    # Adapter before Emulator
    for idx in range(l):
        emulator_and_adapter.append(layers[idx])
    emu_l = l

    # Emulator
    for idx in range(len(emulator)):
        emulator_and_adapter.append(emulator[idx])
    emu_r = emu_l + len(emulator)

    # Adapter after Emulator
    for idx in range(r, len(layers)):
        emulator_and_adapter.append(layers[idx])

    # Need keep raw model when kd applied
    new_model = copy.deepcopy(model)
    new_emulator_and_adapter = copy.deepcopy(emulator_and_adapter)
    # Set student model
    new_model = set_layers(new_model, new_emulator_and_adapter, emu_l, emu_r)

    gc.collect()
    torch.cuda.empty_cache()

    return new_model


def convert_layers_train_state(layers, is_trainable=True):
    """
    Convert the trainability state of a list of layers.

    Args:
        layers (nn.ModuleList): The list of layers to be converted.
        is_trainable (bool): The flag to indicate whether the layers should
            be trainable or not. Default to True.

    Returns:
        None: This function does not return anything, but modifies the
            layers in-place.
    """
    if is_trainable:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


def align_student_with_teacher(raw_model, adap_model, cfg, device, monitor):
    """
    Align the student part of the adapter model with the teacher part using
    knowledge distillation on a held-out dataset.

    Args:
        raw_model (AdapterModel): The original adapter model object that
            contains the causal language model and the adapter layers.
        adap_model (AdapterModel): The compressed adapter model object that
            contains the emulator and the adapter layers.
        cfg (Config): The configuration object that contains the alignment
            parameters.
        device (torch.device): The device to run the alignment on.
        monitor (Monitor): The monitor object to track the FL progress.

    Returns:
        AdapterModel: The aligned adapter model object with the updated
            emulator and adapter layers.
    """
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
                logger.info("Restored the adapter and emulator from ckpt")
                logger.warning(
                    "Please make sure the dtype of model keep the same.")
                # Make student un-trainable
                convert_layers_train_state(adap_model.student,
                                           is_trainable=False)
                does_train_emulator = False
        except Exception as error:
            logger.error(error)

    # Case1: Load ckpt, so we do not need to train student
    if not does_train_emulator:
        return adap_model

    # Case2: Restore fail or not assigned, start to train student
    new_cfg = build_cfg_for_alignment(cfg)

    # Make adapter un-trainable
    convert_layers_train_state(adap_model.adapter, is_trainable=False)

    # Make student trainable
    convert_layers_train_state(adap_model.student, is_trainable=True)

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
    del adap_model.teacher
    adap_model.save_model(cfg.llm.offsite_tuning.emu_align.save_to)

    # Make adapter trainable
    convert_layers_train_state(adap_model.adapter, is_trainable=True)

    # Make student un-trainable
    convert_layers_train_state(adap_model.student, is_trainable=False)

    return adap_model


def wrap_offsite_tuning_for_eval(model, config):
    """
    Wrap the offsite tuning process for evaluation.

    Args:
        model (AdapterModel): The original adapter model object that
            contains the causal language model and the adapter layers.
        config (Config): The configuration object that contains the
            offsite-tuning parameters.

    Returns:
        AdapterModel or nn.Module: The offsite-tuned model object that
            contains the emulator and the adapter layers, or the original model
            object with the adapter layers updated.
    """
    logger.info('===============use offsite tuning===============')
    # We use offsite-tuning in this experiment
    # Use adapter model instead
    compress_strategy = config.llm.offsite_tuning.strategy
    emulator_l = config.llm.offsite_tuning.emu_l
    emulator_r = config.llm.offsite_tuning.emu_r
    offsite_tuning_kwargs = config.llm.offsite_tuning.kwargs[0]
    adap_model = \
        generate_emulator_and_adapter(model,
                                      strategy=compress_strategy,
                                      emulator_l=emulator_l,
                                      emulator_r=emulator_r,
                                      **offsite_tuning_kwargs)
    # Load kd model if ckpt exits
    if config.llm.offsite_tuning.emu_align.use and \
            config.llm.offsite_tuning.eval_type == 'emu':
        if config.llm.offsite_tuning.emu_align.restore_from != '':
            try:
                ckpt = torch.load(
                    config.llm.offsite_tuning.emu_align.restore_from,
                    map_location='cpu',
                )
                adap_model.load_state_dict(ckpt['model'], strict=False)
                logger.info("Restored the adapter and emulator from ckpt")
            except Exception as error:
                logger.warning(error)

    # Load ckpt for eval
    try:
        ckpt = torch.load(config.federate.save_to, map_location='cpu')
        if 'model' and 'cur_round' in ckpt:
            adap_model.load_state_dict(ckpt['model'])
        else:
            adap_model.load_state_dict(ckpt)
    except Exception as error:
        logger.warning(f"{error}, will use raw model.")

    if config.llm.offsite_tuning.eval_type == 'emu':
        model = adap_model
        del model.teacher
    elif config.llm.offsite_tuning.eval_type == 'full':
        # Raw model load adapter from adapter_and_emulator
        new_model_state_dict = model.state_dict()
        for key, value in zip(model.state_dict().keys(),
                              adap_model.state_dict().values()):
            new_model_state_dict[key] = value
        model.load_state_dict(new_model_state_dict, strict=False)
        del adap_model
    else:
        raise NotImplementedError(
            '`config.llm.offsite_tuning.eval_type` should be chosen from '
            '`["emu", "full"]`.')
    return model
