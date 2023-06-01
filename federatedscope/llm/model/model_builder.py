from federatedscope.llm.model.adapter_builder import AdapterModel
import copy

MODEL_CACHE = {}


def get_model_from_huggingface(model_name, config):
    from transformers import AutoModelForCausalLM

    if model_name in MODEL_CACHE:
        model = copy.deepcopy(MODEL_CACHE[model_name])
    else:
        if config.llm.accelerator.use:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", offload_folder="offload")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        MODEL_CACHE[model_name] = model
    return model


def get_model_from_modelscope(model_name, config):
    from modelscope.models import Model

    if model_name in MODEL_CACHE:
        model = copy.deepcopy(MODEL_CACHE[model_name])
    else:
        model = Model.from_pretrained(model_name)
        MODEL_CACHE[model_name] = model
    return model


def get_llm(config):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name, config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)

    return model
