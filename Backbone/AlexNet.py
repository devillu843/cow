import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化

    def forward(self, x):
        #CAM = []
        x = self.features(x)
        #CAM = x
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历每一个层结构，进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏执置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AlexNet_two_part(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False, reduce_channel=False):
        super(AlexNet_two_part, self).__init__()
        self.features = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.reduce_channel = reduce_channel
        if reduce_channel:
            
            self.rc = nn.Sequential(
                nn.Conv2d(256 * 2, 256, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2*256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化
        

    def forward(self, x1, x2):
        #CAM = []
        x1 = self.features(x1)
        x2 = self.features(x2)
        x = torch.cat([x1, x2], dim=1)
        #CAM = x
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        if self.reduce_channel:
            x = self.rc(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历每一个层结构，进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏执置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)





class AlexNet_two_part_alone(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False, reduce_channel=False):
        super(AlexNet_two_part_alone, self).__init__()
        self.features1 = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2 = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.reduce_channel = reduce_channel
        if reduce_channel:
            self.rc = nn.Sequential(
                nn.Conv2d(2*256,256,kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2*256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化

    def forward(self, x1, x2):
        #CAM = []
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x = torch.cat([x1, x2], dim=1)
        if self.reduce_channel:
            x = self.rc(x)
        #CAM = x
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历每一个层结构，进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏执置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AlexNet_three_part(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False, reduce_channel=False):
        super().__init__()
        self.features = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[256, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.reduce_channel = reduce_channel
        if reduce_channel:
            self.rc = nn.Sequential(
                nn.Conv2d(256 * 3, 256, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(3 * 256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化

    def forward(self, x1, x2, x3):
        #CAM = []
        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = self.features(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        if self.reduce_channel:
            x = self.rc(x)
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历每一个层结构，进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏执置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)





class AlexNet_three_part_alone(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False, reduce_channel=False):
        super().__init__()
        self.features1 = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2 = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features3 = nn.Sequential(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.reduce_channel = reduce_channel
        if reduce_channel:
            self.rc = nn.Sequential(
                nn.Conv2d(3*256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(3 * 256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()  # 初始化权重，自动初始化

    def forward(self, x1, x2, x3):
        #CAM = []
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x3 = self.features3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        if self.reduce_channel:
            x = self.rc(x)
        # 展平处理，从1维开始  与x = x.view(-1, 32*5*5)相同
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历每一个层结构，进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏执置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = AlexNet_three_part(num_classes=41).to(device)
    # model = AlexNet_two_part(num_classes=41).to(device)
    # model = AlexNet(num_classes=41).to(device)


    input = torch.randn(1, 3, 224, 224).to(device)
    input2 = torch.randn(1, 3, 224, 224).to(device)
    input3 = torch.randn(1, 3, 224, 224).to(device)

    # summary(model, input_size=[(3, 224, 224)])
    # flops, params = profile(model,inputs=(input,))

    # summary(model, input_size=[(3, 224, 224), (3, 224, 224)])
    # flops, params = profile(model,inputs=(input,input2,))

    summary(model, input_size=[(3, 224, 224), (3, 224, 224),(3, 224, 224)])
    flops, params = profile(model,inputs=(input,input2,input3,))

    print(flops)
    print(params)
