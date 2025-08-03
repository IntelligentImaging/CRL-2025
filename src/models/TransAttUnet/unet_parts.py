""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.image.convolution import SE3Convolution


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 in 3D"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvEQ(nn.Module):
    """(SE3Convolution => [BN] => ReLU) * 2 in 3D"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Define the representation lists for input and output channels
        if in_channels == 1:
            Rs_in = [(in_channels, 0)]
            Rs_out = [(16, 0), (16, 1), (4, 2)]

        if in_channels == 84:
            Rs_in = [(16, 0), (16, 1), (4, 2)]
            Rs_out = [(32, 0), (16, 1), (4, 2)]

        if in_channels == 100:
            Rs_in = [(32, 0), (16, 1), (4, 2)]
            Rs_out = [(64, 0), (16, 1), (4, 2)]

        if in_channels == 132:
            Rs_in = [(64, 0), (16, 1), (4, 2)]
            Rs_out = [(128, 0), (16, 1), (4, 2)]

        Rs_mid = Rs_out

        self.double_conv = nn.Sequential(
            SE3Convolution(Rs_in, Rs_mid, size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            SE3Convolution(Rs_mid, Rs_out, size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv in 3D"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels in [16, 32, 64]:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConvEQ(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv in 3D"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust for 3D
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
