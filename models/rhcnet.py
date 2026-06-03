import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


###############################################
# Channel Shuffle
###############################################
class ChannelShuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):

        b, c, h, w = x.size()
        g = self.groups

        x = x.view(b, g, c // g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)

        return x


###############################################
# Deformable Convolution Block
###############################################
class DeformConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):

        super().__init__()

        padding = kernel // 2
        offset_channels = 2 * kernel * kernel

        self.offset = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=kernel,
            padding=padding
        )

        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding=padding
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        offset = self.offset(x)
        x = self.deform(x, offset)
        x = self.bn(x)

        return x


###############################################
# RHCNet Block
###############################################
class RHCBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        half = channels // 2

        self.regular = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1),
            nn.BatchNorm2d(half),

            nn.Conv2d(half, half, 3, padding=1),
            nn.BatchNorm2d(half),

            nn.ReLU(inplace=True)
        )

        self.def1 = DeformConvBlock(half, half)
        self.def2 = DeformConvBlock(half, half)

        self.relu = nn.ReLU(inplace=True)

        self.shuffle = ChannelShuffle(2)

        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):

        identity = x

        x1, x2 = torch.chunk(x, 2, dim=1)

        r = self.regular(x1)

        d = self.def1(x2)
        d = self.def2(d)
        d = self.relu(d)

        out = torch.cat([r, d], dim=1)

        out = self.shuffle(out)

        out = out + identity

        out = self.norm(out)

        return out
