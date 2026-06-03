import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):

        C1 = 0.01**2
        C2 = 0.03**2

        mu_x = F.avg_pool2d(x,3,1,1)
        mu_y = F.avg_pool2d(y,3,1,1)

        sigma_x = F.avg_pool2d(x*x,3,1,1) - mu_x**2
        sigma_y = F.avg_pool2d(y*y,3,1,1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y,3,1,1) - mu_x*mu_y

        ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))

        return torch.clamp((1 - ssim)/2,0,1).mean()

class TemporalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, prev_frame, gt):

        pred_motion = pred - prev_frame
        gt_motion = gt - prev_frame

        return self.l1(pred_motion, gt_motion)