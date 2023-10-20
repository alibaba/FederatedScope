import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (FloatTensor): batch_size x n_classes
        """
        student_prob = F.log_softmax(output / self.temperature, dim=-1)
        teacher_prob = F.softmax(target / self.temperature, dim=-1)
        return -(teacher_prob * student_prob).mean()
