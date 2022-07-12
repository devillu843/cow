import torch.nn as nn
import torch.nn.functional as F

from .conv_bn import Conv2d_BN, Conv2d_BN_leaky, Conv2d_BN_mish
from .SE_block import SE


class ResNeXt_block(nn.Module):
    def __init__(self, in_channels, cardinality, group_depth, stride, is_se=False) -> None:
        super(ResNeXt_block, self).__init__()
        self.is_se = is_se
        self.group_channels = cardinality * group_depth
        self.conv1 = Conv2d_BN(in_channels, self.group_channels,
                               True, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2d_BN(self.group_channels, self.group_channels,
                               True, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(
            self.group_channels, self.group_channels*2, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_channels*2)
        if self.is_se:
            self.se = SE(self.group_channels*2, 16)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, self.group_channels*2,
                      kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.group_channels*2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn(out)
        if self.is_se:
            co = self.se(out)
            out *= co
        out += self.short_cut(x)
        return F.relu(out)
