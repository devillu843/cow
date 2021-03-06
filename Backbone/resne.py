from turtle import forward
import cv2
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
from myutils.AOLM import AOLM



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        #  downsample=None 短接层
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 先短接在relu
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # channel第一层channel个数
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel,
                      downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_last2 = self.layer4[0:1](x)
        x_last = self.layer4[2](x_last2)

        if self.include_top:
            x = self.avgpool(x_last)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x_last2, x_last, x


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

'''
反正我是对这个模型大小不满意，准备调整下模型大小
'''
class ResNet50_AOLM(nn.Module):
    def __init__(self,block=BasicBlock, blocks_num=[3, 4, 6, 3], num_classes=1000, include_top=True) -> None:
        super().__init__()
        # 返回最后两层特征图
        self.res1 = ResNet(block, blocks_num, num_classes, include_top)
        # 返回最后的特征图，经过flatten和fc进行输出
        self.res2 = ResNet(block, blocks_num, num_classes, include_top)

    def forward(self, x):
        x_last2, x_last, x_linear1 = self.res1(x)

        coordinates = torch.tensor(AOLM(x_last, x_last2))
        batch_size = len(coordinates)
        local_imgs = torch.zeros([batch_size, 3, 480, 480]).to('cuda')  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]

            # interpolate 上下采样函数，调整大小用
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(480, 480),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        _, x_last, x_linear2 = self.res2(local_imgs)
        return x_linear1, x_linear2, x_last




#     Sequential(
#   (0): Bottleneck(
#     (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (downsample): Sequential(
#       (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#       (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (1): Bottleneck(
#     (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#   )
#   (2): Bottleneck(
#     (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#   )
# )


class ALOM_ResNet_two_part(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res1 = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 2, num_classes)

    def forward(self, x1, x2):
        x1_linear,_,x1 = self.res1(x1)
        x1 = self.avgpool(x1)
        x2_linear,_,x2 = self.res1(x2)
        x2 = self.avgpool(x2)
        x = torch.cat([x1,x2], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x1_linear, x2_linear, x

def ALOM_resnet18_two(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def ALOM_resnet34_two(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet50_two(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet101_two(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet152_two(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class ALOM_ResNet_two_part_alone(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res1 = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.res2 = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 2, num_classes)

    def forward(self, x1, x2):
        x1_linear,_,x1 = self.res1(x1)
        x1 = self.avgpool1(x1)
        x2_linear,_,x2 = self.res2(x2)
        x2 = self.avgpool1(x2)
        x = torch.cat([x1,x2], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x1_linear, x2_linear, x



def ALOM_resnet18_two_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part_alone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def ALOM_resnet34_two_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part_alone(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet50_two_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part_alone(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet101_two_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part_alone(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet152_two_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_two_part_alone(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class ALOM_ResNet_three_part(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 3, num_classes)

    def forward(self, x1, x2, x3):
        x1_linear,_,x1 = self.res(x1)
        x1 = self.avgpool(x1)
        x2_linear,_,x2 = self.res(x2)
        x2 = self.avgpool(x2)
        x3_linear,_,x3 = self.res(x3)
        x3 = self.avgpool(x3)
        x = torch.cat([x1,x2,x3], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x1_linear, x2_linear, x3_linear, x


def ALOM_resnet18_three(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def ALOM_resnet34_three(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet50_three(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet101_three(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet152_three(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class ALOM_ResNet_three_part_alone(nn.Module):

    def __init__(self, block, blocks_num, num_classes, include_top) -> None:
        super().__init__()
        self.res1 = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.res2 = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.res3 = ResNet50_AOLM(block, blocks_num, num_classes, include_top)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(block.expansion * 512 * 3, num_classes)

    def forward(self, x1, x2, x3):
        x1_linear,_,x1 = self.res1(x1)
        x1 = self.avgpool1(x1)
        x2_linear,_,x2 = self.res2(x2)
        x2 = self.avgpool2(x2)
        x3_linear,_,x3 = self.res3(x3)
        x3 = self.avgpool3(x3)
        x = torch.cat([x1,x2,x3], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x1_linear, x2_linear, x3_linear, x


def ALOM_resnet18_three_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part_alone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def ALOM_resnet34_three_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part_alone(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet50_three_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part_alone(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet101_three_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part_alone(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def ALOM_resnet152_three_alone(num_classes=1000, include_top=True):
    return ALOM_ResNet_three_part_alone(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)





if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = resnet50(num_classes=41).to(device)
    # model = resnet50_two_alone(num_classes=41).to(device)
    model = resnet50(num_classes=41).to(device)


    input = torch.randn(1, 3, 224, 224).to(device)
    input2 = torch.randn(1, 3, 224, 224).to(device)
    input3 = torch.randn(1, 3, 224, 224).to(device)

    summary(model, input_size=[(3, 224, 224)])
    flops, params = profile(model,inputs=(input,))

    # summary(model, input_size=[(3, 224, 224), (3, 224, 224)])
    # flops, params = profile(model,inputs=(input,input2,))

    # summary(model, input_size=[(3, 224, 224), (3, 224, 224),(3, 224, 224)])
    # flops, params = profile(model,inputs=(input,input2,input3,))

    print(flops)
    print(params)