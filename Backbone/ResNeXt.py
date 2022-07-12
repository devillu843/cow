import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.conv_bn import Conv2d_BN, Conv2d_BN_leaky, Conv2d_BN_mish
from blocks.ResNeXt_block import ResNeXt_block


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, group_depth, num_classes=1000, is_se=False) -> None:
        super(ResNeXt, self).__init__()
        self.is_se = is_se
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = Conv2d_BN(3, self.channels, True,
                               kernel_size=7, stride=2, padding=3)
        group_depth1 = group_depth
        self.conv2 = self._make_layers(group_depth1,num_blocks[0],stride=1)
        group_depth2 = group_depth1 * 2
        self.conv3 = self._make_layers(group_depth2,num_blocks[1],stride=2)
        group_depth3 = group_depth2 * 2
        self.conv4 = self._make_layers(group_depth3,num_blocks[2],stride=2)
        group_depth4 = group_depth3 * 2
        self.conv5 = self._make_layers(group_depth4,num_blocks[3],stride=2)
        self.fc = nn.Linear(self.channels,num_classes) # 224x224 input size
        self.maxpool = nn.MaxPool2d(3,2,1)

    def _make_layers(self, group_depth, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for m in strides:
            layers.append(ResNeXt_block(self.channels, self.cardinality, group_depth, m, self.is_se))
            self.channels = self.cardinality * group_depth * 2
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x,3,2,1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

def resNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes)


def resNeXt101_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 32, 4, num_classes)


def resNeXt101_64x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 64, 4, num_classes)


def resNeXt50_32x4d_SE(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes, is_se=True)

def test():
    x = torch.randn(1, 3, 224, 224)
    net = resNeXt50_32x4d(num_classes=80)
    y = net(x)
    print(y.size())

if __name__ == '__main__':

    test()

    