import torch.nn as nn

'''维度转换'''
# class MultiConv(nn.Module):
#     def __init__(self, in_ch, out_ch, attn=True):
#         super(MultiConv, self).__init__()
#
#         self.fuse_attn = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.PReLU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.PReLU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=1),
#             nn.BatchNorm2d(out_ch),
#             nn.Softmax2d() if attn else nn.PReLU()
#         )
#
#     def forward(self, x):
#         return self.fuse_attn(x)
class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch, attn=True):
        super(MultiConv, self).__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm3d(out_ch),
            nn.Softmax(dim=1) if attn else nn.PReLU()
        )

    def forward(self, x):
        return self.fuse_attn(x)
