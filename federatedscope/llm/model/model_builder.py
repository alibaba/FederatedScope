import copy

MODEL_CACHE = {}


def enable_adapter(model, adapter, package, **kwargs):
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        Support methods:
            LoRA
            Prefix Tuning
            P-Tuning
            Prompt Tuning
            AdaLoRA
        """
        # from peft import get_peft_model, TaskType
        #
        # config = getattr(import_module('peft'), f'{adapter}Config')
        # peft_config = config(task_type=TaskType.SEQ_2_SEQ_LM, **kwargs)
        # model = get_peft_model(model, peft_config)

        raise NotImplementedError

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        Support methods:
            Bottleneck Adapters
            Prefix Tuning
            LoRA
            Compacter
            Adapter Fusion
            Invertible Adapters
            Parallel block
        """
        raise NotImplementedError

    return model


def get_model_from_huggingface(model_name):
    from transformers import AutoModelForCausalLM

    if model_name in MODEL_CACHE:
        model = copy.deepcopy(MODEL_CACHE[model_name])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        MODEL_CACHE[model_name] = model
    return model


def get_model_from_modelscope(model_name):
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
        model = get_model_from_huggingface(model_name=model_name)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name)
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

    return model
