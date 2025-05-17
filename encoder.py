import torch.nn as nn
import torch
from nets.speatt import speatt
from torchsummary import summary
from timm.models.layers import to_2tuple
from nets.MSA import LiteMLA


class IRBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, size=None, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
            self.spe = speatt(size)

    def forward(self, x):
        if self.use_res_connect:
            spe = self.spe(x)
            return spe + self.conv(x)
        else:
            return self.conv(x)

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-3):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = LiteMLA(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path = nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(10, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down =  IRBlock(64, 64, 2, 128)
        self.layer1 = nn.Sequential(
            IRBlock(64, 64, 1, 128),
            IRBlock(64, 64, 1, 128)
        )
        self.layer2 = nn.Sequential(
            IRBlock(64, 128, 2, 64),
            IRBlock(128, 128, 1, 64)
        )
        self.layer3 = nn.Sequential(
            IRBlock(128, 256, 2, 32),
            IRBlock(256, 256, 1, 32)
        )
        self.layer4 = nn.Sequential(
            IRBlock(256, 512, 2, 16),
            IRBlock(512, 512, 1, 16),
            MSABlock(512),
            MSABlock(512),
            MSABlock(512),
            MSABlock(512)
        )

    def forward(self, x):
        feat1 = self.prepare(x)
        x = self.down(feat1)

        feat2 = self.layer1(x)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


if __name__ == '__main__':
    # 定义测试模型
    device = torch.device('cuda')
    model = Encoder().to(device)
    # 打印模型结构
    # print(model)

    summary(model, input_size=(10, 512, 512))
