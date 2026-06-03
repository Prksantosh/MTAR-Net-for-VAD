
import torch.nn as nn
from models.aerhcnet import AE_RHCNet

###############################################
# Decoder Stage
###############################################
class DecoderStage(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.aerhc = AE_RHCNet(in_ch)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.aerhc(x)

        x = self.up(x)

        return x

