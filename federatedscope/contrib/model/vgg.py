import torch
import torch.nn as nn

from federatedscope.register import register_model

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    ],
    'VGG16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'VGG19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel=3, num_classe=10):
    return VGG('VGG11', channel=channel, num_classes=num_classe)


def VGG13(channel=3, num_classe=10):
    return VGG('VGG13', channel=channel, num_classes=num_classe)


def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)


def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


def vgg(model_config):
    if '11' in model_config.type:
        net = VGG11()
    elif '13' in model_config.type:
        net = VGG13()
    return net


def call_vgg(model_config, local_data):
    if 'vgg' in model_config.type:
        model = vgg(model_config)
        return model


register_model('vgg', call_vgg)
