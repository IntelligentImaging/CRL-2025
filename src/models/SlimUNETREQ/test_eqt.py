import torch
import torch.nn as nn
from se3cnn.image.convolution import SE3Convolution
import torch.nn.functional as F
from se3cnn.image.gated_block import GatedBlock

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#
#
# # Define a simple SE3Convolution layer
# class SimpleSE3Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(SimpleSE3Conv, self).__init__()
#         self.conv = SE3Convolution([(1, 0)], [(2, 0), (2, 1), (2, 2)], size=5)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # Initialize the layer
# se3_conv = SimpleSE3Conv(1, 1)  # Adjust the channels as needed
#
# # Create a random 3D input tensor
# input_tensor = torch.randn(1, 1, 96, 96, 96)  # Batch size 1, Channel 1, 10x10x10 grid
#
# # Pass the input through the SE3Convolution layer
# output_tensor = se3_conv(input_tensor)
#
# print(output_tensor.shape)
#

n_in = 1
chan_config = [
    [16, 16, 4],
    [16, 16, 4],
    [16, 16, 4],
    [16, 16, 4]
]

features = [[n_in]] + chan_config + [[64]]
block_params = [{'activation': F.relu}] * (len(features) - 2) + [{'activation': F.relu}]

common_block_params = {
    'size': 5,
    'stride': 1,
    'padding': 2,
    'normalization': None,
    'capsule_dropout_p': None,
    'smooth_stride': False,
}
blocks = [
    GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i])
    for i in range(len(block_params))
]

print(blocks)

# Create a random 3D input tensor
input_tensor = torch.randn(1, 1, 96, 96, 96)  # Batch size 1, Channel 1, 10x10x10 grid

# Pass the input through the SE3Convolution layer
output_tensor = blocks[0](input_tensor)
print(output_tensor.shape)

output_tensor = blocks[1](output_tensor)
print(output_tensor.shape)

output_tensor = blocks[2](output_tensor)
print(output_tensor.shape)

output_tensor = blocks[3](output_tensor)
print(output_tensor.shape)

output_tensor = blocks[4](output_tensor)
print(output_tensor.shape)