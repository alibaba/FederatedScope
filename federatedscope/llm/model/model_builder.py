import copy
# MODEL_CACHE = {}
from federatedscope.llm.model.model_adapter import AdapterModel


def enable_adapter(model, adapter, package, **kwargs):
    adapter = adapter.lower()
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
        from peft import get_peft_model
        if adapter == 'lora':
            # //TODO: fix error
            from peft import LoraConfig
            peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
        # elif adapter == 'prefix':
        #     from peft import PrefixEncoder, PrefixTuningConfig
        #
        #     peft_config = PrefixTuningConfig(
        #         peft_type="PREFIX_TUNING",
        #         task_type="SEQ_2_SEQ_LM",
        #         num_virtual_tokens=20,
        #         token_dim=768,
        #         num_transformer_submodules=1,
        #         num_attention_heads=12,
        #         num_layers=12,
        #         encoder_hidden_size=768,
        #     )
        #     model = get_peft_model(model, peft_config)
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")

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
        # TODO: maybe we need to merge the following adapters rightly
        if adapter == 'lora':
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            from transformers.adapters import AdapterConfig, ConfigUnion

            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"There is no adapter named {adapter} in {package}")
    else:
        raise NotImplementedError
    return model


def get_model_from_huggingface(model_name):
    from transformers import AutoModelForCausalLM

    # if model_name in MODEL_CACHE:
    #     model = copy.deepcopy(MODEL_CACHE[model_name])
    # else:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #     MODEL_CACHE[model_name] = model
    return model


def get_model_from_modelscope(model_name):
    from modelscope.models import Model

    # if model_name in MODEL_CACHE:
    #     model = copy.deepcopy(MODEL_CACHE[model_name])
    # else:
    model = Model.from_pretrained(model_name)
    #     MODEL_CACHE[model_name] = model
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

    use_adapter = config.adapter.use
    if use_adapter:
        adapter_package = config.adapter.args[0]['adapter_package']
        adapter_method = config.adapter.args[0]['adapter_method']
        model = enable_adapter(model, adapter_method, adapter_package)

    model = AdapterModel(model, use_adapter)

    return model
