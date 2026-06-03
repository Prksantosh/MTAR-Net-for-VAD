import torch.nn as nn
from models.encoder import EncoderStage
from models.decoder import DecoderStage
from models.timestamp import TimestampTransform
#from models.mema_e3d_emu import MemE3DLSTM
#from models.mem_2d_lstm import MemE3DLSTM
#from models.mem_GALSTM import MemE3DLSTM
from models.convlstm import ConvLSTM

###############################################
# Full Autoencoder Model
###############################################
class RHCNetAutoencoder(nn.Module):

    def __init__(self, seq_len=3):

        super().__init__()

        self.seq_len = seq_len

        # Initial feature extraction
        self.initial = nn.Conv2d(3, 24, 3, padding=1)

        # Encoder
        self.enc1 = EncoderStage(24, 48)
        self.enc2 = EncoderStage(48, 96)
        self.enc3 = EncoderStage(96, 192)
        self.enc4 = EncoderStage(192, 384)

        # Timestamp transformation
        self.timestamp = TimestampTransform()

        # Temporal modeling
        #self.e3d = MemE3DLSTM(384,384)   
        self.e3d = ConvLSTM(384,384)

        # Decoder
        self.dec1 = DecoderStage(384, 192)
        self.dec2 = DecoderStage(192, 96)
        self.dec3 = DecoderStage(96, 48)
        self.dec4 = DecoderStage(48, 24)

        # Final reconstruction
        self.final = nn.Conv2d(24, 3, 3, padding=1)


    def forward(self, x):

        B, T, C, H, W = x.shape

        # --------------------------------
        # Merge batch and time
        # --------------------------------
        x = x.view(B * T, C, H, W)

        # --------------------------------
        # Initial feature extraction
        # --------------------------------
        x = self.initial(x)

        # --------------------------------
        # Encoder
        # --------------------------------
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # --------------------------------
        # Convert to spatiotemporal tensor
        # (B,C,T,H,W)
        # --------------------------------
        x = self.timestamp(x, B, T)

        # --------------------------------
        # E3D-LSTM Temporal Modeling
        # --------------------------------
        # --------------------------------
        # Convert (B,C,T,H,W) -> (B,T,C,H,W)
        # --------------------------------
        x = x.permute(0, 2, 1, 3, 4)
        
        # --------------------------------
        # ConvLSTM Temporal Modeling
        # --------------------------------
        x, _ = self.e3d(x)
        
        # x : (B,T,C,H,W)
        
        # Take last temporal feature
        x = x[:, -1]
        # --------------------------------
        # Decoder
        # --------------------------------
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        # --------------------------------
        # Final output frame
        # --------------------------------
        x = self.final(x)

        return x
