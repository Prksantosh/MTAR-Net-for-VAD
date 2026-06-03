import torch.nn as nn
from losses.losses import SSIMLoss, TemporalLoss



class Config:
    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    temporal_loss = TemporalLoss()
    
 

    device = "cuda"

