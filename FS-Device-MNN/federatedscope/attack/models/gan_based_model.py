import torch
import torch.nn as nn
from copy import deepcopy


class GeneratorFemnist(nn.Module):
    '''
    The generator for Femnist dataset
    '''
    def __init__(self, noise_dim=100):
        super(GeneratorFemnist, self).__init__()

        module_list = []
        module_list.append(
            nn.Linear(in_features=noise_dim,
                      out_features=4 * 4 * 256,
                      bias=False))
        module_list.append(nn.BatchNorm1d(num_features=4 * 4 * 256))
        module_list.append(nn.ReLU())
        self.body1 = nn.Sequential(*module_list)

        # need to reshape the output of self.body1

        module_list = []

        module_list.append(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               bias=False))
        module_list.append(nn.BatchNorm2d(128))
        module_list.append(nn.ReLU())
        self.body2 = nn.Sequential(*module_list)

        module_list = []
        module_list.append(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               bias=False))
        module_list.append(nn.BatchNorm2d(64))
        module_list.append(nn.ReLU())
        self.body3 = nn.Sequential(*module_list)

        module_list = []
        module_list.append(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=1,
                               kernel_size=(4, 4),
                               stride=(2, 2),
                               bias=False))
        module_list.append(nn.BatchNorm2d(1))
        module_list.append(nn.Tanh())
        self.body4 = nn.Sequential(*module_list)

    def forward(self, x):

        tmp1 = self.body1(x).view(-1, 256, 4, 4)

        assert tmp1.size()[1:4] == (256, 4, 4)

        tmp2 = self.body2(tmp1)
        assert tmp2.size()[1:4] == (128, 6, 6)

        tmp3 = self.body3(tmp2)

        assert tmp3.size()[1:4] == (64, 13, 13)

        tmp4 = self.body4(tmp3)
        assert tmp4.size()[1:4] == (1, 28, 28)

        return tmp4
