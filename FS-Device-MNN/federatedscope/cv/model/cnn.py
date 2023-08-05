import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU

from torch.nn import init
import torch
import math


class ConvNet2(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=1024,
                 class_num=10):
        super(ConvNet2, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ConvNet2Body(Module):
    def __init__(self,
                 in_channels,
                 h=28,
                 w=28,
                 hidden=1024,
                 class_num=62):
        super(ConvNet2Body, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = self.relu(self.fc1(x))

        return x


class ConvNet5(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=512,
                 class_num=2):
        super(ConvNet5, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        # self.bn1 = BatchNorm2d(32)

        self.conv2 = Conv2d(32, 64, 5, padding=2)
        # self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(64, 64, 5, padding=2)
        # self.bn3 = BatchNorm2d(64)

        self.conv4 = Conv2d(64, 128, 5, padding=2)
        # self.bn4 = BatchNorm2d(128)

        self.conv5 = Conv2d(128, 128, 5, padding=2)
        # self.bn5 = BatchNorm2d(128)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 128,
            hidden)
        self.fc2 = Linear(hidden, class_num)

    def forward(self, x):
        # x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        # x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        # x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        # x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)

        # x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ConvNet5Body(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=512,
                 class_num=10):
        super(ConvNet5Body, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.bn1 = BatchNorm2d(32)

        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.bn3 = BatchNorm2d(64)

        self.conv4 = Conv2d(64, 128, 5, padding=2)
        self.bn4 = BatchNorm2d(128)

        self.conv5 = Conv2d(128, 128, 5, padding=2)
        self.bn5 = BatchNorm2d(128)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 128,
            hidden)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = self.relu(self.fc1(x))

        return x


class VGG11(Module):
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=128,
                 class_num=10,
                 dropout=.0):
        super(VGG11, self).__init__()

        self.conv1 = Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = BatchNorm2d(64)

        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = BatchNorm2d(128)

        self.conv3 = Conv2d(128, 256, 3, padding=1)
        self.bn3 = BatchNorm2d(256)

        self.conv4 = Conv2d(256, 256, 3, padding=1)
        self.bn4 = BatchNorm2d(256)

        self.conv5 = Conv2d(256, 512, 3, padding=1)
        self.bn5 = BatchNorm2d(512)

        self.conv6 = Conv2d(512, 512, 3, padding=1)
        self.bn6 = BatchNorm2d(512)

        self.conv7 = Conv2d(512, 512, 3, padding=1)
        self.bn7 = BatchNorm2d(512)

        self.conv8 = Conv2d(512, 512, 3, padding=1)
        self.bn8 = BatchNorm2d(512)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 512,
            hidden)
        self.fc2 = Linear(hidden, hidden)
        self.fc3 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn7(self.conv7(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn8(self.conv8(x)))
        x = self.maxpool(x)

        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x
