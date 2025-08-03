import sys
import os
current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_path)
sys.path.append(parent_dir)

from TransAttUnet.unet_parts import *
from TransAttUnet.unet_parts_att_transformer import *
from TransAttUnet.unet_parts_att_multiscale import *


class UNet_Attention_Transformer_Multiscale(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Attention_Transformer_Multiscale, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvEQ(n_channels, 84)
        self.down1 = Down(84, 100)
        self.down2 = Down(100, 132)
        self.down3 = Down(132, 196)
        self.down4 = Down(196, 256)
        factor = 2 if bilinear else 1
        self.down5 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(580, 128 // factor, bilinear)
        self.up3 = Up(324, 64 // factor, bilinear)
        self.up4 = Up(196, 32 // factor, bilinear)
        self.up5 = Up(132, 16, bilinear)
        self.outc = OutConv(32, n_classes)

        '''位置编码'''
        self.pos = PositionEmbeddingLearned(256 // factor)

        '''空间注意力机制'''
        self.pam = PAM_Module(256)

        '''自注意力机制'''
        self.sdpa = ScaledDotProductAttention(256)

        '''残差多尺度连接'''
        self.fuse1 = MultiConv(768, 256)
        self.fuse2 = MultiConv(384, 128)
        self.fuse3 = MultiConv(192, 64)
        self.fuse4 = MultiConv(128, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        '''Setting 1'''
        x6_pam = self.pam(x6)

        '''Setting 2'''
        x6_pos = self.pos(x6)
        x6 = x6 + x6_pos
        x6_sdpa = self.sdpa(x6)
        x6 = x6_sdpa + x6_pam

        x7 = self.up1(x6, x5)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='trilinear', align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), 1)

        x8 = self.up2(x7_cat, x4)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='trilinear', align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), 1)

        x9 = self.up3(x8_cat, x3)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='trilinear', align_corners=True)
        x9_cat = torch.cat((x8_scale, x9), 1)

        x10 = self.up4(x9_cat, x2)
        x9_scale = F.interpolate(x9, size=x10.shape[2:], mode='trilinear', align_corners=True)
        x10_cat = torch.cat((x9_scale, x10), 1)

        x11 = self.up5(x10_cat, x1)
        x10_scale = F.interpolate(x10, size=x11.shape[2:], mode='trilinear', align_corners=True)
        x11 = torch.cat((x10_scale, x11), 1)

        logits = self.outc(x11)

        return logits


if __name__ == "__main__":
    x = torch.randn(size=(1, 1, 128, 128, 128))
    model = UNet_Attention_Transformer_Multiscale(1, 2, bilinear=True)
    y = model(x)
    print(y.shape)

