import torch
import torch.nn as nn
from nets.MSA_block import MSA
from nets.mv2 import InvertedResidual
from nets.dysample import DySample
from nets.mobilevit import mobile_vit_small

class Updecoder(nn.Module):
    def __init__(self, in_size, out_size, input2_c):
        super(Updecoder, self).__init__()
        self.conv1 = InvertedResidual(
            in_size, out_size, stride=1, expand_ratio=2)
        self.conv2 = InvertedResidual(
            out_size, out_size, stride=1, expand_ratio=2)
        self.up = DySample(input2_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class MVMSANet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(MVMSANet, self).__init__()
        in_filters = [160, 320, 608, 768]
        out_filters = [64, 128, 256, 512]
        self.mobilevit = mobile_vit_small(pretrained=pretrained)
        self.up_concat4 = Updecoder(in_filters[3], out_filters[3], 640)
        self.up_concat3 = Updecoder(in_filters[2], out_filters[2], 512)
        self.up_concat2 = Updecoder(in_filters[1], out_filters[1], 256)
        self.up_concat1 = Updecoder(in_filters[0], out_filters[0], 128)
        self.f1mssa = MSA(32, 32,map_reduce=1)
        self.f2mssa = MSA(64, 64,map_reduce=1)
        self.f3mssa = MSA(96, 96,map_reduce=4)
        self.f4mssa = MSA(128, 128,map_reduce=6)
        self.f5mssa = MSA(640, 640,map_reduce=8)

        self.up_conv = nn.Sequential(
                DySample(64),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0],
                          kernel_size=3, padding=1),
                nn.ReLU(),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4,feat5] = self.mobilevit.forward(inputs)
        feat1 = self.f1mssa(feat1)
        feat2 = self.f2mssa(feat2)
        feat3 = self.f3mssa(feat3)
        feat4 = self.f4mssa(feat4)
        feat5 = self.f5mssa(feat5)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
   
        if self.up_conv != None:
            up1 = self.up_conv(up1)
        final = self.final(up1)
        return final

    def freeze_backbone(self):
        for param in self.mobilevit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.mobilevit.parameters():
            param.requires_grad = True
