import torch.nn as nn
import torch.nn.functional


class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True,**kwargs) -> None:
        super(Conv2d_BN, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, **kwargs),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Conv2d_BN_leaky(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv2d_BN, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, **kwargs),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.LeakyReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Conv2d_BN_mish(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv2d_BN, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, **kwargs),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.Mish(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
