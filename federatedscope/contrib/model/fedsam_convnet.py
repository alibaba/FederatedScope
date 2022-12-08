'''The implementation of ASAM and SAM are borrowed from
    https://github.com/debcaldarola/fedsam
   Caldarola, D., Caputo, B., & Ciccone, M.
   Improving Generalization in Federated Learning by Seeking Flat Minima,
   European Conference on Computer Vision (ECCV) 2022.
'''
import os
import re
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from federatedscope.register import register_model


class Conv2Model(nn.Module):
    def __init__(self, num_classes):
        super(Conv2Model, self).__init__()
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.classifier = nn.Sequential(nn.Linear(64 * 5 * 5, 384), nn.ReLU(),
                                        nn.Linear(384, 192), nn.ReLU(),
                                        nn.Linear(192, self.num_classes))

        self.size = self.model_size()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size


def call_fedsam_conv2(model_config, local_data):
    if model_config.type == 'fedsam_conv2':
        model = Conv2Model(10)
        return model


register_model('fedsam_conv2', call_fedsam_conv2)
