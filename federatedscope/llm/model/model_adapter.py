import torch
import torch.nn as nn


class AdapterModel(nn.Module):
    def __init__(self, model, use_adapter=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.base_model = model
        self.use_adapter = use_adapter

    def forward(self, *args, **kwargs):
        return self.base_model.forward(*args, **kwargs)

    def state_dict(self):
        if self.use_adapter:
            return self.get_trainable_state_dict(self.base_model)
        else:
            return self.base_model.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        if self.use_adapter:
            return self.base_model.load_state_dict(state_dict, strict=False)
        else:
            return self.base_model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self, model):
        return {k: v for k, v in model.named_parameters() if v.requires_grad}
