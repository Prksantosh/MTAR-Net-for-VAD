# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:03:30 2026

@author: USER
"""

import torch
import torch.nn as nn

from models.encoder import EncoderStage
from models.decoder import DecoderStage
from models.timestamp import TimestampTransform
from models.mem_GALSTM import MemE3DLSTM
#from models.mem_2d_lstm import MemE3DLSTM
#from models.mema_e3d_emu import MemE3DLSTM


class RHCNetAutoencoder(nn.Module):

    def __init__(self, seq_len=3):

        super().__init__()

        self.seq_len = seq_len



        self.initial = nn.Conv2d(
            3,
            24,
            3,
            padding=1
        )

 

        self.enc1 = EncoderStage(24, 48)
        self.enc2 = EncoderStage(48, 96)
        self.enc3 = EncoderStage(96, 192)
        self.enc4 = EncoderStage(192, 384)


        self.timestamp = TimestampTransform()



        self.e3d = MemE3DLSTM(384, 384)



        self.dec1 = DecoderStage(384, 192)
        self.dec2 = DecoderStage(192, 96)
        self.dec3 = DecoderStage(96, 48)
        self.dec4 = DecoderStage(48, 24)



        self.skip1 = nn.Conv2d(192 + 192, 192, 1)
        self.skip2 = nn.Conv2d(96 + 96, 96, 1)
        self.skip3 = nn.Conv2d(48 + 48, 48, 1)
        self.skip4 = nn.Conv2d(24 + 24, 24, 1)



        self.final = nn.Conv2d(
            24,
            3,
            3,
            padding=1
        )

    def forward(self, x):

        B, T, C, H, W = x.shape



        x = x.view(B * T, C, H, W)



        x0 = self.initial(x)



        e1 = self.enc1(x0)   
        e2 = self.enc2(e1)  
        e3 = self.enc3(e2)  
        e4 = self.enc4(e3)  



        x = self.timestamp(e4, B, T)



        x = self.e3d(x)


        x = x[:, :, -1] 



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



        x = self.dec1(x)              

        x = torch.cat(
            [x, e3_last],
            dim=1
        )

        x = self.skip1(x)



        x = self.dec2(x)             

        x = torch.cat(
            [x, e2_last],
            dim=1
        )

        x = self.skip2(x)



        x = self.dec3(x)             

        x = torch.cat(
            [x, e1_last],
            dim=1
        )

        x = self.skip3(x)



        x = self.dec4(x)             

        x = torch.cat(
            [x, x0_last],
            dim=1
        )

        x = self.skip4(x)



        x = self.final(x)

        return x
