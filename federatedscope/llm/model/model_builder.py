from federatedscope.llm.model.adapter_builder import AdapterModel


def get_model_from_huggingface(model_name):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


def get_llm(config):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name)
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

    use_adapter = config.llm.adapter.use
    if use_adapter:
        adapter_package = config.llm.adapter.args[0]['adapter_package']
        adapter_method = config.llm.adapter.args[0]['adapter_method']
    else:
        adapter_package = None
        adapter_method = None

    model = AdapterModel(model, use_adapter, adapter_package, adapter_method)

    return model
