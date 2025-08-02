import torch
from torch import nn
from torch.nn import functional as F
# from prettytable import PrettyTable


def get_conv_transform(in_channels, out_channels, mode):
    """
    :param in_channels: int
    :param out_channels: int
    :param mode: string in ['up, 'down', 'same']
        'up' - ConvTranspose3d with output 2*D, 2*W, 2*H
        'down' - Conv3d with output D/2, W/2, H/2
        'same' - Conv3d with output D, W, H
        where D, W, H - shape of the input
    :return:
    """
    if mode == 'up':
        return nn.ConvTranspose3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=2,
                                  stride=2,
                                  padding=0)
    elif mode == 'down':
        return nn.Conv3d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         stride=2,
                         padding=1)
    elif mode == 'same':
        return nn.Conv3d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=1,
                         stride=1)