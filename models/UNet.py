from collections import OrderedDict
import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_multiplier: int = 1,
    ):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features1 = 64
        self.features2 = self.features1 * 2  # 128
        self.features3 = self.features2 * 2  # 256
        self.features4 = self.features3 * 2  # 512

        self.encoder1 = self._enc_block(
            self.in_channels,
            self.features1,
            "enc1",
        )
        self.encoder2 = self._enc_block(
            self.features1 * feature_multiplier,
            self.features2,
            "enc2",
        )
        self.encoder3 = self._enc_block(
            self.features2 * feature_multiplier,
            self.features3,
            "enc3",
        )
        self.encoder4 = self._enc_block(
            self.features3 * feature_multiplier,
            self.features4,
            "enc4",
        )
        self.encoder5 = self._enc_block(
            self.features4 * feature_multiplier,
            self.features4,
            "enc5",
        )
        self.encoder6 = self._enc_block(
            self.features4 * feature_multiplier,
            self.features4,
            "enc6",
        )
        self.encoder7 = self._enc_block(
            self.features4 * feature_multiplier,
            self.features4,
            "enc7",
        )
        self.encoder8 = self._enc_block(
            self.features4 * feature_multiplier,
            self.features4,
            "enc8",
        )

        self.encoder_m = self._enc_block(
            self.features4 * feature_multiplier,
            self.features4,
            "encm",
        )
        self.decoder_m = self._dec_block(
            self.features4 * feature_multiplier,
            self.features4,
            "decm",
        )

        self.decoder8 = self._dec_block(
            self.features4 * (2 + int(feature_multiplier / 2)),
            self.features4,
            "dec8",
        )
        self.decoder7 = self._dec_block(
            self.features4 * (2 + int(feature_multiplier / 2)),
            self.features4,
            "dec7",
        )
        self.decoder6 = self._dec_block(
            self.features4 * (2 + int(feature_multiplier / 2)),
            self.features4,
            "dec6",
        )
        self.decoder5 = self._dec_block(
            self.features4 * (2 + int(feature_multiplier / 2)),
            self.features4,
            "dec5",
        )
        self.decoder4 = self._dec_block(
            self.features4 * (2 + int(feature_multiplier / 2)),
            self.features3,
            "dec4",
        )
        self.decoder3 = self._dec_block(
            self.features3 * (2 + int(feature_multiplier / 2)),
            self.features2,
            "dec3",
        )
        self.decoder2 = self._dec_block(
            self.features2 * (2 + int(feature_multiplier / 2)),
            self.features1,
            "dec2",
        )
        self.decoder1 = self._dec_block(
            self.features1 * (2 + int(feature_multiplier / 2)),
            self.out_channels,
            "dec1",
        )

    def _enc_block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv",
                        nn.Conv2d(
                            in_channels,
                            features,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=features)),
                    (name + "LReLU", nn.LeakyReLU(0.2)),
                ]
            )
        )

    def _dec_block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "tconv",
                        nn.ConvTranspose2d(
                            in_channels,
                            features,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    (name + "norm", nn.BatchNorm2d(num_features=features)),
                    (name + "LReLU", nn.LeakyReLU(0.2)),
                ]
            )
        )

    def forward(self, x):
        # x = inx512x512
        x1 = self.encoder1(x)  # 64x256x256
        x2 = self.encoder2(x1)  # 128x128x128
        x3 = self.encoder3(x2)  # 256x64x64
        x4 = self.encoder4(x3)  # 512x32x32
        x5 = self.encoder5(x4)  # 512x16x16
        x6 = self.encoder6(x5)  # 512x8x8
        x7 = self.encoder7(x6)  # 512x4x4
        x8 = self.encoder8(x7)  # 512x2x2

        m = self.encoder_m(x8)  # 512x1x1
        y = self.decoder_m(m)  # 512x2x2

        y = torch.cat((y, x8), dim=1)  # 1024x2x2
        y = self.decoder8(y)  # 512x4x4
        y = torch.cat((y, x7), dim=1)  # 1024x4x4
        y = self.decoder7(y)  # 512x8x8
        y = torch.cat((y, x6), dim=1)  # 1024x8x8
        y = self.decoder6(y)  # 512x16x16
        y = torch.cat((y, x5), dim=1)  # 1024x16x16
        y = self.decoder5(y)  # 512x32x32
        y = torch.cat((y, x4), dim=1)  # 1024x32x32
        y = self.decoder4(y)  # 256x64x64
        y = torch.cat((y, x3), dim=1)  # 512x64x64
        y = self.decoder3(y)  # 128x128x128
        y = torch.cat((y, x2), dim=1)  # 256x128x128
        y = self.decoder2(y)  # 64x256x256
        y = torch.cat((y, x1), dim=1)  # 128x256x256
        y = self.decoder1(y)  # outx512x512
        return y
