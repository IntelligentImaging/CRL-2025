import sys
import os

# Get the directory of the current script
current_path = os.path.dirname(os.path.realpath(__file__))

# Append the directory to sys.path
sys.path.append(current_path)

from se3cnn.image.convolution import SE3Convolution
from se3cnn.image.gated_block import GatedBlock
import torch
import torch.nn as nn

from Slim_UNETR_Block import Block


class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DepthwiseConvLayer, self).__init__()

        # features = [(dim_in,),
        #             (16, 2, 2),  # 32 channels | 4(1)+2(3)+2(5) = 4+6+10 =32
        #             (32, 4, 4),  # 64 channels | 4(1)+4(3)+4(5) = 4+12+20= 64
        #             (64, 8, 8),  # 128 channels | 8(1)+8(3)+8(5) = 8+24+40=72
        #             (32, 4, 4),
        #             (16, 2, 2),
        #             (dim_out,)]

        self.depth_wise = GatedBlock(dim_in, dim_out, size=r, stride=r, normalization="group", padding=0)
        # self.point_wise = GatedBlock(dim_in, dim_out, size=1, stride=1, normalization="group", padding=0)
        # self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        # x = self.point_wise(x)
        # x = self.norm(x)
        return x

# class DepthwiseConvLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, r):
#         super(DepthwiseConvLayer, self).__init__()
#         self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r)
#         self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
#
#     def forward(self, x):
#         x = self.depth_wise(x)
#         x = self.norm(x)
#         return x

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels=4,
            embed_dim=396,
            embedding_dim=27,
            channels=(36, 108, 216),
            blocks=(1, 2, 3, 2),
            heads=(1, 2, 4, 8),
            r=(4, 2, 2, 1),
            dropout=0.3,
    ):
        # channels = [(4, 4, 4),  # 32 channels | 4(1)+2(3)+2(5) = 4+6+10 =32
        #             (8, 8, 4),  # 64 channels | 4(1)+4(3)+4(5) = 4+12+20= 64
        #             (16, 16, 4),  # 128 channels | 8(1)+8(3)+8(5) = 8+24+40=72
        #             # (32, 4, 4),
        #             # (16, 2, 2),
        #             # (1,)
        #             ]

        super(Encoder, self).__init__()
        self.DWconv1 = DepthwiseConvLayer(dim_in=(1,), dim_out=(4, 4, 4), r=r[0])
        self.DWconv2 = DepthwiseConvLayer(dim_in=(4, 4, 4), dim_out=(8, 8, 4), r=r[1])
        self.DWconv3 = DepthwiseConvLayer(dim_in=(8, 8, 4), dim_out=(16, 16, 4), r=r[2])
        self.DWconv4 = DepthwiseConvLayer(dim_in=(16, 16, 4), dim_out=(embed_dim,), r=r[3])
        # self.DWconv1 = DepthwiseConvLayer(dim_in=in_channels, dim_out=channels[0], r=4)
        # self.DWconv2 = DepthwiseConvLayer(dim_in=channels[0], dim_out=channels[1], r=2)
        # self.DWconv3 = DepthwiseConvLayer(dim_in=channels[1], dim_out=channels[2], r=2)
        # self.DWconv4 = DepthwiseConvLayer(dim_in=channels[2], dim_out=embed_dim, r=2)
        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, embedding_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states_out = []
        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        B, C, W, H, Z = x.shape
        x = self.block4(x)
        x = x.flatten(2).transpose(-1, -2)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, hidden_states_out, (B, C, W, H, Z)
