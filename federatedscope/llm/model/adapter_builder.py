import torch.nn as nn
import copy
from collections import OrderedDict


def enable_adapter(model, package, adapter, **kwargs):
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
            from peft import LoraConfig
            peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)

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

            # TODO: configure these args in cfg
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


class AdapterModel(nn.Module):
    def __init__(self,
                 model,
                 use_adapter=False,
                 adapter_package=None,
                 adapter_method=None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.base_model = model
        self.use_adapter = use_adapter
        self.adapter_package = adapter_package
        self.adapter_method = adapter_method
        if self.use_adapter:
            self.adapted_model = enable_adapter(self.base_model,
                                                self.adapter_package,
                                                self.adapter_method)
        else:
            self.adapted_model = self.base_model

    def forward(self, *args, **kwargs):
        return self.adapted_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.adapted_model.generate(*args, **kwargs)

    def state_dict(self):
        if self.use_adapter:
            return self.get_trainable_state_dict(self.adapted_model)
        else:
            return self.adapted_model.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        if self.use_adapter:
            return self.adapted_model.load_state_dict(state_dict, strict=False)
        else:
            return self.adapted_model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self, model):
        grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        model_state_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict
