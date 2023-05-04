from collections import OrderedDict
import torch
from torch import nn

from .UNet import UNet


class DoubleUNet(nn.Module):
    def __init__(self, input1_channels: int = 13, input2_channels: int = 3):
        super(DoubleUNet, self).__init__()
        self.unet1 = UNet(input2_channels, input1_channels)
        self.unet2 = UNet(
            input1_channels, input2_channels, feature_multiplier=2
        )

    def forward(self, u, v):
        # --- UNet 1
        xu1 = self.unet1.encoder1(u)  # 64x256x256
        xu2 = self.unet1.encoder2(xu1)  # 128x128x128
        xu3 = self.unet1.encoder3(xu2)  # 256x64x64
        xu4 = self.unet1.encoder4(xu3)  # 512x32x32
        xu5 = self.unet1.encoder5(xu4)  # 512x16x16
        xu6 = self.unet1.encoder6(xu5)  # 512x8x8
        xu7 = self.unet1.encoder7(xu6)  # 512x4x4
        xu8 = self.unet1.encoder8(xu7)  # 512x2x2
        mu = self.unet1.encoder_m(xu8)  # 512x1x1
        yu8 = self.unet1.decoder_m(mu)  # 512x2x2
        yu = torch.cat((yu8, xu8), dim=1)  # 1024x2x2
        yu7 = self.unet1.decoder8(yu)  # 512x4x4
        yu = torch.cat((yu7, xu7), dim=1)  # 1024x4x4
        yu6 = self.unet1.decoder7(yu)  # 512x8x8
        yu = torch.cat((yu6, xu6), dim=1)  # 1024x8x8
        yu5 = self.unet1.decoder6(yu)  # 512x16x16
        yu = torch.cat((yu5, xu5), dim=1)  # 1024x16x16
        yu4 = self.unet1.decoder5(yu)  # 512x32x32
        yu = torch.cat((yu4, xu4), dim=1)  # 1024x32x32
        yu3 = self.unet1.decoder4(yu)  # 256x64x64
        yu = torch.cat((yu3, xu3), dim=1)  # 512x64x64
        yu2 = self.unet1.decoder3(yu)  # 128x128x128
        yu = torch.cat((yu2, xu2), dim=1)  # 256x128x128
        yu1 = self.unet1.decoder2(yu)  # 64x256x256
        yu = torch.cat((yu1, xu1), dim=1)  # 128x256x256
        yu = self.unet1.decoder1(yu)  # outx512x512

        # --- UNet 2
        xv1 = self.unet2.encoder1(v)  # 64x256x256
        xv = torch.cat((xv1, yu1), dim=1)  # 128x256x256
        xv2 = self.unet2.encoder2(xv)  # 128x128x128
        xv = torch.cat((xv2, yu2), dim=1)  # 256x128x128
        xv3 = self.unet2.encoder3(xv)  # 256x64x64
        xv = torch.cat((xv3, yu3), dim=1)  # 512x64x64
        xv4 = self.unet2.encoder4(xv)  # 512x32x32
        xv = torch.cat((xv4, yu4), dim=1)  # 1024x32x32
        xv5 = self.unet2.encoder5(xv)  # 512x16x16
        xv = torch.cat((xv5, yu5), dim=1)  # 1024x16x16
        xv6 = self.unet2.encoder6(xv)  # 512x8x8
        xv = torch.cat((xv6, yu6), dim=1)  # 1024x8x8
        xv7 = self.unet2.encoder7(xv)  # 512x4x4
        xv = torch.cat((xv7, yu7), dim=1)  # 1024x4x4
        xv8 = self.unet2.encoder8(xv)  # 512x2x2
        xv = torch.cat((xv8, yu8), dim=1)  # 1024x2x2
        mv = self.unet2.encoder_m(xv)  # 512x1x1
        xv = torch.cat((mv, mu), dim=1)
        yv8 = self.unet2.decoder_m(xv)
        yv = torch.cat((yv8, xv8, yu8), dim=1)
        yv7 = self.unet2.decoder8(yv)
        yv = torch.cat((yv7, xv7, yu7), dim=1)
        yv6 = self.unet2.decoder7(yv)
        yv = torch.cat((yv6, xv6, yu6), dim=1)
        yv5 = self.unet2.decoder6(yv)
        yv = torch.cat((yv5, xv5, yu5), dim=1)
        yv4 = self.unet2.decoder5(yv)
        yv = torch.cat((yv4, xv4, yu4), dim=1)
        yv3 = self.unet2.decoder4(yv)
        yv = torch.cat((yv3, xv3, yu3), dim=1)
        yv2 = self.unet2.decoder3(yv)
        yv = torch.cat((yv2, xv2, yu2), dim=1)
        yv1 = self.unet2.decoder2(yv)
        yv = torch.cat((yv1, xv1, yu1), dim=1)
        yv = self.unet2.decoder1(yv)

        return yu, yv
