import copy

from transformers import AutoModelForCausalLM

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


def get_model_from_huggingface(model_name, llm_config, **kwargs):
    if model_name in MODEL_CACHE:
        model = copy.deepcopy(MODEL_CACHE[model_name])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        MODEL_CACHE[model_name] = model
    model.resize_token_embeddings(llm_config.tok_len)
    return model


def get_model_from_modelscope(model_name, llm_config, **kwargs):
    import modelscope

    if model_name in MODEL_CACHE:
        model = copy.deepcopy(MODEL_CACHE[model_name])
    else:
        model = modelscope.load(model_name)
        MODEL_CACHE[model_name] = model
    # model.resize_token_embeddings(llm_config.tok_len)
    return model


def get_model(model_config, **kwargs):
    model_name, model_hub = model_config.type.split('@')
    llm_config = model_config.llm

    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           llm_config=llm_config)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name,
                                          llm_config=llm_config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')
    return model


if __name__ == '__main__':
    # Test cases
    from federatedscope.core.configs.config import CN

    llm_config = CN()
    llm_config.tok_len = 128

    model = get_model_from_huggingface(model_name='gpt2',
                                       llm_config=llm_config)
