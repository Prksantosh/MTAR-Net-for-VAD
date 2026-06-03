# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:02:23 2026

@author: USER
"""

import torch
import torch.nn as nn

from models.encoder import EncoderStage
from models.decoder import DecoderStage
from models.timestamp import TimestampTransform
from models.convlstm import ConvLSTM


###############################################
# Full Autoencoder Model with Skip Connections
###############################################
class RHCNetAutoencoder(nn.Module):

    def __init__(self, seq_len=3):

        super().__init__()

        self.seq_len = seq_len

        #################################################
        # Initial Feature Extraction
        #################################################

        self.initial = nn.Conv2d(
            3,
            24,
            kernel_size=3,
            padding=1
        )

        #################################################
        # Encoder
        #################################################

        self.enc1 = EncoderStage(24, 48)
        self.enc2 = EncoderStage(48, 96)
        self.enc3 = EncoderStage(96, 192)
        self.enc4 = EncoderStage(192, 384)

        #################################################
        # Timestamp Transformation
        #################################################

        self.timestamp = TimestampTransform()

        #################################################
        # Temporal Modeling
        #################################################

        self.e3d = ConvLSTM(384, 384)

        #################################################
        # Decoder
        #################################################

        self.dec1 = DecoderStage(384, 192)
        self.dec2 = DecoderStage(192, 96)
        self.dec3 = DecoderStage(96, 48)
        self.dec4 = DecoderStage(48, 24)

        #################################################
        # Skip Fusion Layers
        #################################################

        self.skip1 = nn.Sequential(

            nn.Conv2d(
                192 + 192,
                192,
                kernel_size=1
            ),

            nn.BatchNorm2d(192),

            nn.SiLU(inplace=True)
        )

        self.skip2 = nn.Sequential(

            nn.Conv2d(
                96 + 96,
                96,
                kernel_size=1
            ),

            nn.BatchNorm2d(96),

            nn.SiLU(inplace=True)
        )

        self.skip3 = nn.Sequential(

            nn.Conv2d(
                48 + 48,
                48,
                kernel_size=1
            ),

            nn.BatchNorm2d(48),

            nn.SiLU(inplace=True)
        )

        self.skip4 = nn.Sequential(

            nn.Conv2d(
                24 + 24,
                24,
                kernel_size=1
            ),

            nn.BatchNorm2d(24),

            nn.SiLU(inplace=True)
        )

        #################################################
        # Final Reconstruction
        #################################################

        self.final = nn.Conv2d(
            24,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        """
        x: [B, T, C, H, W]
        """

        B, T, C, H, W = x.shape

        #################################################
        # Merge Batch + Time
        #################################################

        x = x.view(B * T, C, H, W)

        #################################################
        # Initial Features
        #################################################

        x0 = self.initial(x)

        #################################################
        # Encoder
        #################################################

        e1 = self.enc1(x0)   # [BT,48,H/2,W/2]

        e2 = self.enc2(e1)   # [BT,96,H/4,W/4]

        e3 = self.enc3(e2)   # [BT,192,H/8,W/8]

        e4 = self.enc4(e3)   # [BT,384,H/16,W/16]

        #################################################
        # Convert to Spatiotemporal Tensor
        # (B,C,T,H,W)
        #################################################

        x = self.timestamp(e4, B, T)

        #################################################
        # ConvLSTM Temporal Modeling
        #################################################
        # Convert:
        # (B,C,T,H,W) -> (B,T,C,H,W)
        #################################################

        x = x.permute(0, 2, 1, 3, 4)

        x, _ = self.e3d(x)

        #################################################
        # Last Temporal Feature
        #################################################

        x = x[:, -1]   # [B,384,H/16,W/16]

        #################################################
        # Last Frame Skip Features
        #################################################

        e3_last = (
            e3.view(
                B,
                T,
                192,
                e3.shape[-2],
                e3.shape[-1]
            )[:, -1]
        )

        e2_last = (
            e2.view(
                B,
                T,
                96,
                e2.shape[-2],
                e2.shape[-1]
            )[:, -1]
        )

        e1_last = (
            e1.view(
                B,
                T,
                48,
                e1.shape[-2],
                e1.shape[-1]
            )[:, -1]
        )

        x0_last = (
            x0.view(
                B,
                T,
                24,
                x0.shape[-2],
                x0.shape[-1]
            )[:, -1]
        )

        #################################################
        # Decoder Stage 1
        #################################################

        x = self.dec1(x)

        x = torch.cat(
            [x, e3_last],
            dim=1
        )

        x = self.skip1(x)

        #################################################
        # Decoder Stage 2
        #################################################

        x = self.dec2(x)

        x = torch.cat(
            [x, e2_last],
            dim=1
        )

        x = self.skip2(x)

        #################################################
        # Decoder Stage 3
        #################################################

        x = self.dec3(x)

        x = torch.cat(
            [x, e1_last],
            dim=1
        )

        x = self.skip3(x)

        #################################################
        # Decoder Stage 4
        #################################################

        x = self.dec4(x)

        x = torch.cat(
            [x, x0_last],
            dim=1
        )

        x = self.skip4(x)

        #################################################
        # Final Reconstruction
        #################################################

        x = self.final(x)

        return x