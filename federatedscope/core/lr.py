import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_channels, class_num, use_bias=True):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Linear(in_channels, class_num, bias=use_bias)

    def forward(self, x):
        return self.fc(x)
