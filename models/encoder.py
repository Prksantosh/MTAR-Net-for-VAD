import torch.nn as nn
from models.rhcnet import RHCBlock

###############################################
# Encoder Stage
###############################################
class EncoderStage(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.rhc = RHCBlock(in_ch)

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.rhc(x)
        x = self.down(x)

        return x


