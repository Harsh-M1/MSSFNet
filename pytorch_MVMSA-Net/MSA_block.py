import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out+avg_out)
        return output*x
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(
                out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MSA(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(MSA, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = BasicConv(in_planes, 2*inter_planes, kernel_size=1,
                              stride=1, groups=groups, relu=False)
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1,
                      stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3),
                      stride=stride, padding=(1, 1), groups=groups),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1,
                      padding=3, dilation=3, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1,
                      padding=1, dilation=1, relu=False, groups=groups)
        )
        self.branch2_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1,
                      padding=5, dilation=5, relu=False, groups=groups)
        )
        self.ca = CALayer(5 * 2 * inter_planes)
        self.ConvLinear = BasicConv(
            5 * 2 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(
            in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(x)
        xap = F.interpolate(image_features, size=size, mode='bilinear')
        b1 = self.branch1(x)
        b1_1 = self.branch1_1(b1)
        b2 = self.branch2(b1)
        b2_1 = self.branch2_1(b2)
        out = torch.cat((xap, b1, b2, b1_1, b2_1), 1)
        out = self.ca(out)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out + short
        out = self.relu(out)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mssa = MSA(160, 640).to(device)
    summary(mssa, input_size=(160, 16, 16))