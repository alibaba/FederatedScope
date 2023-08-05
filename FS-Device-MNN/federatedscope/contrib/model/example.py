from federatedscope.register import register_model


# Build you torch or tf model class here
from federatedscope.register import register_model
import torch
from torch.nn import Flatten
from torch.nn import MaxPool2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU

class LeNet(torch.nn.Module):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(LeNet, self).__init__()

        self.body = torch.nn.Sequential(
            Conv2d(1, 20, kernel_size=5, padding=5//2, stride=2),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(980, 200),
            ReLU(),
            Linear(200, self.n_classes)
        )

    def forward(self, x):
        return self.body(x)

# Instantiate your model class with config and data
def ModelBuilder(n_classes):

    model = LeNet(n_classes)

    return model


def call_my_net(model_config, local_data):
    if model_config.type.lower() == "lenet10":
        model = ModelBuilder(n_classes=10)
        return model
    elif model_config.type.lower() == "lenet62":
        model = ModelBuilder(n_classes=62)
        return model


register_model("lenet10", call_my_net)
register_model("lenet62", call_my_net)
