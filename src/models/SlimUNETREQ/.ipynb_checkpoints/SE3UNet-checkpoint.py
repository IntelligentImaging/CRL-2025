import torch
import torch.nn as nn
import torch.nn.functional as F
from se3cnn.image.gated_block import GatedBlock


class Merge(nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


# Define the SE3 U-Net with GatedBlock
class SE3UNet(nn.Module):
    def __init__(self, output_size, filter_size=3):
        super(SE3UNet, self).__init__()

        size = filter_size
        output_feature = (output_size,)
        features = [(1, 0, 0, 0),  # 1
                    (16, 0, 0, 0),
                    (32, 0, 0, 0),
                    (64, 0, 0, 0),
                    (128, 0, 0, 0), ]
        # (8, 2, 2, 0),  # 24
        # (16, 4, 4, 0),  # 48
        # (32, 8, 8, 0),  # 96
        # (64, 16, 16, 0)]  # 192

        # Encoder 1
        self.conv1 = nn.Sequential(
            GatedBlock(features[0], features[1], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15),
            GatedBlock(features[1], features[1], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15))

        # Encoder 2
        self.conv2 = nn.Sequential(
            GatedBlock(features[1], features[2], size=size, padding=size // 2, stride=2, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15),
            GatedBlock(features[2], features[2], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15))

        # Encoder 3
        self.conv3 = nn.Sequential(
            GatedBlock(features[2], features[3], size=size, padding=size // 2, stride=2, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15),
            GatedBlock(features[3], features[3], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15))

        # Bottleneck
        self.conv4 = nn.Sequential(
            GatedBlock(features[3], features[4], size=size, padding=size // 2, stride=2, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15),
            GatedBlock(features[4], features[4], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch", capsule_dropout_p=0.15))

        # Decoder 1
        # self.up1 = GatedBlock(features[4], features[3], size=size, padding=size // 2, stride=2,
        #                       activation=(F.relu, F.sigmoid),
        #                       normalization="batch", transpose=True)
        self.up1 = torch.nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)

        self.merge1 = Merge()

        self.conv5 = nn.Sequential(
            GatedBlock(features[4], features[3], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch"),
            GatedBlock(features[3], features[3], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch"))

        # Decoder 2
        # self.up2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode="nearest"),
        #     GatedBlock(features[3], features[2], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
        #                normalization="batch"))

        self.up2 = torch.nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)

        self.merge2 = Merge()

        self.conv6 = nn.Sequential(
            GatedBlock(features[3], features[2], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch"),
            GatedBlock(features[2], features[2], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch"))

        # Decoder 3
        # self.up3 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode="nearest"),
        #     GatedBlock(features[2], features[1], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
        #                normalization="batch"))

        self.up3 = torch.nn.ConvTranspose3d(32, 16, 3, 2, 1, 1)

        self.merge3 = Merge()

        self.conv7 = nn.Sequential(
            GatedBlock(features[2], features[1], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch"),
            GatedBlock(features[1], features[1], size=size, padding=size // 2, stride=1, activation=(F.relu, F.sigmoid),
                       normalization="batch"))

        # Segmentation Head
        self.conv_final = GatedBlock(features[1], output_feature, size=1, padding=0, stride=1)

    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        # Bottleneck
        Bottleneck_out = self.conv4(conv3_out)

        # Decoder
        up1_out = self.up1(Bottleneck_out)
        merge1_out = self.merge1(conv3_out, up1_out)
        conv5_out = self.conv5(merge1_out)
        up2_out = self.up2(conv5_out)
        merge2_out = self.merge2(conv2_out, up2_out)
        conv6_out = self.conv6(merge2_out)
        up3_out = self.up3(conv6_out)
        merge3_out = self.merge3(conv1_out, up3_out)
        conv7_out = self.conv7(merge3_out)

        # Segmentation Head
        out = self.conv_final(conv7_out)

        return out


# Model initialization
model = SE3UNet(output_size=2)

# Dummy input for testing
dummy_input = torch.rand(1, 1, 96, 96, 96)  # Example input shape
output = model(dummy_input)

print(output.shape)
