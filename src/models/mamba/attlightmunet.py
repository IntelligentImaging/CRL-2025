from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba


def get_dwconv_layer(spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False):
    depth_conv = Convolution(
        spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
        strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels
    )
    point_conv = Convolution(
        spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
        strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1
    )
    return nn.Sequential(depth_conv, point_conv)


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, norm: tuple | str, kernel_size: int = 3, act: tuple | str = ("RELU", {"inplace": True})):
        super().__init__()
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(spatial_dims, in_channels, in_channels)
        self.conv2 = get_mamba_layer(spatial_dims, in_channels, in_channels)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + identity


class ResUpBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, norm: tuple | str, kernel_size: int = 3, act: tuple | str = ("RELU", {"inplace": True})):
        super().__init__()
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(spatial_dims, in_channels, in_channels, kernel_size=kernel_size)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels_g: int, in_channels_x: int, inter_channels: int):
        super().__init__()
        self.W_g = Convolution(spatial_dims=spatial_dims, in_channels=in_channels_g, out_channels=inter_channels, kernel_size=1, act=None, norm=None)
        self.W_x = Convolution(spatial_dims=spatial_dims, in_channels=in_channels_x, out_channels=inter_channels, kernel_size=1, act=None, norm=None)
        self.psi = Convolution(spatial_dims=spatial_dims, in_channels=inter_channels, out_channels=1, kernel_size=1, act=("SIGMOID", {}), norm=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class LightMUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()

        n_up = len(self.blocks_up)
        self.att_blocks = nn.ModuleList()
        for i in range(n_up):
            gating_ch = self.init_filters * 2 ** (n_up - i) // 2
            skip_ch = gating_ch
            inter_ch = gating_ch // 2
            self.att_blocks.append(AttentionBlock(spatial_dims, gating_ch, skip_ch, inter_ch))

        self.conv_final = self._make_final_conv(out_channels)
        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.blocks_down):
            in_ch = self.init_filters * 2**i
            downsample_mamba = get_mamba_layer(self.spatial_dims, in_ch // 2, in_ch, stride=2) if i > 0 else nn.Identity()
            blocks = [ResMambaBlock(self.spatial_dims, in_ch, norm=self.norm, act=self.act) for _ in range(num_blocks)]
            layers.append(nn.Sequential(downsample_mamba, *blocks))
        return layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        n_up = len(self.blocks_up)
        for i, num_blocks in enumerate(self.blocks_up):
            sample_in_channels = self.init_filters * 2 ** (n_up - i)
            block_list = [ResUpBlock(self.spatial_dims, sample_in_channels // 2, norm=self.norm, act=self.act) for _ in range(num_blocks)]
            up_layers.append(nn.Sequential(*block_list))
            up_samples.append(nn.Sequential(
                get_conv_layer(self.spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                get_upsample_layer(self.spatial_dims, sample_in_channels // 2, upsample_mode=self.upsample_mode),
            ))
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]):
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            gating = up(x)
            skip = down_x[i + 1]
            skip_att = self.att_blocks[i](gating, skip)
            x = gating + skip_att
            x = upl(x)
        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor):
        x, down_x = self.encode(x)
        down_x.reverse()
        x = self.decode(x, down_x)
        return x


act = ("RELU", {"inplace": True})
norm = ("INSTANCE", {"affine": True})

def ATTLightMUNetModel(spatial_dims, in_channels, out_channels, feature_size, dropout, global_skip=False, pretrained=False, **kwargs):
    return LightMUNet(
        spatial_dims=spatial_dims,
        init_filters=feature_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        act=act,
        norm=norm
    )

# from thop import profile
# Instantiate the model
# model = ATTLightMUNetModel(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=18,
#     feature_size=24,
#     dropout=0.0
# ).cuda()

# # Create a random 3D input tensor [batch=1, channels=1, depth=128, height=128, width=128]
# x = torch.randn(1, 1, 128, 128, 128).cuda()
#
# # Forward pass
# out = model(x)
# print("Output shape:", out.shape)
# #
# # # Profile the model to get MACs and parameter count
# # macs, params = profile(model, inputs=(x,))
# # print("MACs (multiply-accumulate operations):", macs)
# # print("Number of parameters:", params)