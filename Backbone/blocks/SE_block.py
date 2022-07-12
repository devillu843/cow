from turtle import forward
from matplotlib.pyplot import cla
import torch.nn as nn


class SE(nn.Module):
    def __init__(self, in_channels, ratio) -> None:
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(
            in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0)
        self.excitation = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.compress(x)
        x = nn.ReLU(x)
        x = self.excitation(x)
        return nn.Sigmoid(x)
