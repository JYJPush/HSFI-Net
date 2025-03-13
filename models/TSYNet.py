import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SwinTransformer import SwinTransformer3D


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, k, s, p, g=1, d=(1, 1, 1), bias=False):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
        self.norm = nn.GroupNorm(32, out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)

        x = self.norm(x)
        x = self.relu(x)
        return x


class CTFC(nn.Module):
    def __init__(self, channels, T):
        super(CTFC, self).__init__()
        self.T = T
        self.branch1 = nn.Sequential(
            BasicConv3d(channels, channels // 3, k=(3, 1, 1), s=(1, 1, 1), p=(1, 0, 0)),
            BasicConv3d(channels // 3, channels // 3, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(channels, channels // 3, k=(5, 1, 1), s=(1, 1, 1), p=(2, 0, 0)),
            BasicConv3d(channels // 3, channels // 3, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(channels, channels // 3, k=(7, 1, 1), s=(1, 1, 1), p=(3, 0, 0)),
            BasicConv3d(channels // 3, channels // 3, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
        )

        self.conv = BasicConv3d(channels, channels, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))

    def forward(self, x, y):
        CCT = []
        for i in range(0, self.T):
            xi = x[:, :, i, :, :].unsqueeze(2)
            yi = y[:, :, i, :, :].unsqueeze(2)
            xy = torch.cat((xi, yi), dim=2)
            CCT.append(xy)
        z = CCT[0]
        for i in range(1, self.T):
            z = torch.cat((z, CCT[i]), dim=2)
        z1 = self.branch1(z)
        z2 = self.branch2(z)
        z3 = self.branch3(z)

        out = self.conv(torch.cat((z1, z2, z3), 1))
        return out


class TFF(nn.Module):
    def __init__(self, channels, m_channel):
        super(TFF, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv3d(channels, m_channel, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            BasicConv3d(channels, m_channel, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1)),

        )
        self.branch3 = nn.Sequential(
            BasicConv3d(channels, m_channel, k=(3, 1, 1), s=(1, 1, 1), p=(1, 0, 0)),
            BasicConv3d(m_channel, m_channel, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
        )
        self.branch4 = nn.Sequential(
            BasicConv3d(channels, m_channel, k=(5, 1, 1), s=(1, 1, 1), p=(2, 0, 0)),
            BasicConv3d(m_channel, m_channel, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
        )
        self.conv1 = BasicConv3d(m_channel, m_channel, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv2 = BasicConv3d(m_channel, m_channel, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        xm = self.conv1(x1 + x2 + x3 + x4)
        out = self.conv2(xm)
        return out


class SFE(nn.Module):
    def __init__(self, channels, m_channel):
        super(SFE, self).__init__()
        self.C = m_channel // 4
        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.conv1 = BasicConv3d(channels, m_channel, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1))

        self.conv2 = BasicConv3d(m_channel, m_channel, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0))
        self.conv3 = BasicConv3d(m_channel // 4, m_channel // 4, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv4 = BasicConv3d(m_channel // 4, m_channel // 4, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv5 = BasicConv3d(m_channel // 4, m_channel // 4, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv6 = BasicConv3d(m_channel, m_channel, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0))

        self.conv7 = nn.Sequential(
            BasicConv3d(m_channel, m_channel, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1)),
            self.upsampling2
        )
        self.conv8 = nn.Sequential(
            BasicConv3d(m_channel, m_channel, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
            self.upsampling2
        )

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        y1, y2, y3, y4 = y[:, 0:self.C, :, :, :], y[:, self.C:self.C*2, :, :, :], y[:, self.C*2:self.C*3, :, :, :], y[:, self.C*3:self.C*4, :, :, :]
        t1 = y1
        t2 = self.conv3(y1 + y2)
        t3 = self.conv4(t2 + y3)
        t4 = self.conv5(t3 + y4)
        t = self.conv6(torch.cat((t1, t2, t3, t4), 1)) + x
        z = self.conv7(t)
        out = self.conv8(z)
        return out


class FFM(nn.Module):
    def __init__(self, channels):
        super(FFM, self).__init__()
        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.conv1 = BasicConv3d(channels*4, channels, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.fc1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Sigmoid(),
        )
        self.conv2 = BasicConv3d(channels, channels, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1))
        self.conv3 = nn.Sequential(
            BasicConv3d(channels, 64, k=(2, 3, 3), s=(2, 1, 1), p=(0, 1, 1)),
            self.upsampling2,
            BasicConv3d(64, 32, k=(1, 3, 3), s=(1, 1, 1), p=(0, 1, 1)),
            self.upsampling2,
            nn.Conv3d(32, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        y = self.conv1(torch.cat((x1, x2, x3, x4), 1))
        SG = self.fc1(x4)
        ye = self.conv2(y * SG + y)
        out = self.conv3(ye)

        return out


class TSYNet(nn.Module):
    def __init__(self, T=32, h=224, w=384, pretrained=None):
        super(TSYNet, self).__init__()
        self.VSTbackbone = SwinTransformer3D(pretrained=pretrained)
        self.Conv_DT1 = BasicConv3d(96, 192, k=(2, 1, 1), s=(2, 1, 1), p=(0, 0, 0))
        self.Conv_DT2 = BasicConv3d(192, 192, k=(2, 1, 1), s=(2, 1, 1), p=(0, 0, 0))
        self.Conv_DT3 = BasicConv3d(384, 192, k=(2, 1, 1), s=(2, 1, 1), p=(0, 0, 0))
        self.Conv_DT4 = BasicConv3d(768, 192, k=(2, 1, 1), s=(2, 1, 1), p=(0, 0, 0))

        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.upsampling16 = nn.Upsample(scale_factor=(1, 16, 16), mode='trilinear', align_corners=False)

        self.CTFC3 = CTFC(192, 8)   # MFI unit
        self.CTFC2 = CTFC(192, 8)
        self.CTFC1 = CTFC(192, 8)

        self.TFF1 = TFF(192, 128)   # TFE unit
        self.TFF2 = TFF(192, 128)
        self.SFE3 = SFE(192, 128)   # CFE unit
        self.SFE4 = SFE(192, 128)

        self.out_module = FFM(128)  # SSP module

    def forward(self, x):
        [x1, x2, x3, x4] = self.VSTbackbone(x)
        x1 = self.Conv_DT1(x1)
        x2 = self.Conv_DT2(x2)
        x3 = self.Conv_DT3(x3)
        x4 = self.Conv_DT4(x4)

        t4 = x4
        t3 = self.CTFC3(x3, self.upsampling2(t4))
        t2 = self.CTFC2(x2, self.upsampling2(t3))
        t1 = self.CTFC1(x1, self.upsampling2(t2))

        y1 = self.TFF1(t1)
        y2 = self.TFF2(self.upsampling2(t2))
        y3 = self.SFE3(t3)
        y4 = self.SFE4(self.upsampling2(t4))

        out = self.out_module(y1, y2, y3, y4)
        out = out.view(out.size(0), out.size(3), out.size(4))

        return out


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    image = torch.randn(1, 3, 32, 224, 384).to(device)
    model = TSYNet(pretrained=None).to(device)
    x = model(image)
    print(x.shape)