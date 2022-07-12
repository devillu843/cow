from io import IncrementalNewlineDecoder
import torch
from torch import nn
from torch.nn.modules import activation

'''
没啥注释可以写的，都是网络结构，去看网络结构图
'''


class SiLU(nn.Module):
    @staticmethod  # 静态参数，网络不一定需要传入参数
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act='silu') -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels,
                             kernel_size, stride, act=act)

    def forward(self, x):
        #-----------------------------------------------------#
        #   inchannels = 3,out_channels=64,kernel_size = 3,act = silu
        #   (batch_size,3,640,640) -> (batch_size,12,320,320)
        #-----------------------------------------------------#
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left,
                      patch_top_right, patch_bot_right), dim=1)
        #-----------------------------------------------------#
        #   in_channels = 3,out_channels=64,kernel_size = 3,act = silu
        #   经过BaseConv
        #   (batch_size,12,320,320) -> (batchsize,64,320,320)
        #-----------------------------------------------------#
        return self.conv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act='silu') -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act='silu') -> None:
        super().__init__()
        self.dconv = BaseConv(in_channels, out_channels,
                              kernel_size, stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels,
                              1, stride, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act='silu') -> None:
        super().__init__()
        #-------------------------------------------------------#
        #   dark5
        #   1024,1024,act=silu,hidden=512,conv2_channels = 2048
        #-------------------------------------------------------#
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)  # 2048
        self.conv2 = BaseConv(
            conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        #-------------------------------------------------------#
        #   (batchsize,1024,20,20) -> (batchsize,2048,20,20) -> (batchsize,2048,20,20) -> (batchsize,1024,20,20)
        #-------------------------------------------------------#
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act='silu') -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        #----------------------------------------------------#
        #   dark2: (64,64,n=3,shortcut=True,expansion=1.0,depthwise=False,act=silu) hidden=64
        #   (batchsize,64,160,160) -> (batchsize,64,160,160) -> (batchsize,64,160,160)
        #   self.use_add = True
        #   (batchsize,64,160,160) -> (batchsize,64,160,160)
        #----------------------------------------------------#
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu') -> None:
        # ch_in,ch_out,number,shortcut,groups,expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv3 = BaseConv(2 * hidden_channels,
                              out_channels, 1, stride=1, act=act)

        module_list = [Bottleneck(hidden_channels, hidden_channels,
                                  shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        #----------------------------------------------------#
        #   dark2: (128,128,n=3,depthwise=False,act=silu) hidden=64
        #   (batchsize,128,160,160) -> x1:(batchsize,64,160,160),x2:(batchsize,64,160,160)
        #   m: x1:(batchsize,64,160,160) -> (batchsize,64,160,160) -> (batchsize,64,160,160) -> (batchsize,64,160,160)
        #   cat: x:(batchsize,128,160,160)
        #   return: (batchsize,128,160,160) -> (batchsize,128,160,160)
        #----------------------------------------------------#
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.m(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), depthwise=False, act='silu') -> None:
        super().__init__()
        '''
        dep_mul:默认为1.0
        wid_mul:默认为1.0

        '''
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, kernel_size=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            # (64,128,ks=3,strides=2,act=silu)
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2,
                     n=base_depth, depthwise=depthwise, act=act)  # (128,128,n=3,depthwise=False,act=silu)
        )

        # dark3
        self.dark3 = nn.Sequential(
            # (128,256,ks=3,strides=2,act=silu)
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4,
                     n=base_depth * 3, depthwise=depthwise, act=act)  # (256,256,n=9,depthwise=False,act=silu)
        )

        # dark4
        self.dark4 = nn.Sequential(
            # (256,512,ks=3,strides=2,act=silu)
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8,
                     n=base_depth * 3, depthwise=depthwise, act=act)  # (512,512,n=9,depthwise=False,act=silu)
        )

        # dark5
        self.dark5 = nn.Sequential(
            # (512,1024,ks=3,strides=2,act=silu)
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16,
                          base_channels * 16, act=act),
            CSPLayer(base_channels * 16, base_channels * 16,
                     n=base_depth, shortcut=False, depthwise=depthwise, act=act)  # (1024,1024,n=3,shortcut=False,depthwise=False,act=silu)
        )

    def forward(self, x):
        outputs = {}
        # 先经过Focus网络，(batchsize,3,640,640) -> (batchsize,64,320,320)
        x = self.stem(x)
        outputs["stem"] = x
        # 经过dark2，(batchsize,64,320,320) -> (batchsize,128,160,160) -> (batchsize,128,160,160)
        x = self.dark2(x)
        outputs["dark2"] = x
        # 经过dark3,(batchsize,128,160,160) -> (batchsize,256,80,80) -> (batchsize,256,80,80)
        x = self.dark3(x)
        outputs["dark3"] = x
        # 经过dark4,(batchsize,256,80,80) -> (batchsize,512,40,40) -> (batchsize,512,40,40)
        x = self.dark4(x)
        outputs["dark4"] = x
        # 经过dark5,(batchsize,512,40,40) -> (batchsize,1024,20,20) -> (batchsize,1024,20,20) -> (batchsize,1024,20,20)
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
