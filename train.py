import os
import torch
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from configs.config import Config
from models.autoencoder_skip import RHCNetAutoencoder
from losses.losses import CombinedPredictionLoss
#from losses.losses import SSIMLoss, TemporalLoss
from datasets.avenue_dataset import Avenuedataset, AvenueValDataset
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score



# --------------------------------------------------
# Config
# --------------------------------------------------
config = Config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

save_dir = "checkpoints_ST"
os.makedirs(save_dir, exist_ok=True)

best_model_path = os.path.join(save_dir, "best_model.pth")
last_model_path = os.path.join(save_dir, "last_model.pth")
checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

resume_training = True

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = RHCNetAutoencoder(seq_len=3).to(device)


criterion = CombinedPredictionLoss(
    lambda_mse=0.50,
    lambda_ssim=0.15,
    lambda_temp=0.25,
    lambda_grad=0.20
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)


# --------------------------------------------------
# Transforms
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    #transforms.Normalize(
        #mean=[0.5, 0.5, 0.5],
        #std=[0.5, 0.5, 0.5]
    #)
])


# --------------------------------------------------
# Dataset + Dataloader
# --------------------------------------------------

train_dataset = Avenuedataset(
    root_dir=r"C:\Users\USER\Desktop\Results MGTT\Updated kip model\data\Shanghai_train",
    seq_len=3,
    transform=transform
)

anomaly_ranges_dict = {
    "01_0014": [(148, 240)]

}

val_dataset = AvenueValDataset(
    root_dir=r"C:\Users\USER\Desktop\MVA\eidetic_vad-main\eidetic_vad-main_Shanghaitech\data\Shanghai_val",
    seq_len=3,
    transform=transform,
    anomaly_ranges_dict=anomaly_ranges_dict,
    one_based=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)



# --------------------------------------------------
# Resume training if checkpoint exists
# --------------------------------------------------
start_epoch = 0
best_auc = 0.0

if resume_training and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_auc = checkpoint["best_auc"]
    print(f"Resumed from epoch {start_epoch} | Best AUC: {best_auc:.4f}")


# --------------------------------------------------
# Validation / AUC evaluation
# --------------------------------------------------


def compute_psnr(pred, target, eps=1e-8):
    """
    pred, target: (B, C, H, W), range [0,1]
    returns: (B,)
    """
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=(1, 2, 3)) + eps

    psnr = 10 * torch.log10(1.0 / mse)
    return psnr


def compute_ssim_score(ssim_loss):
    """
    Assuming your criterion returns:
    ssim_loss = (1 - SSIM)
    """
    ssim = 1.0 - ssim_loss
    return ssim


def normalize_tensor(x):
    """
    Min-max normalization per batch
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def validate(model, val_loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_temp = 0.0
    total_grad = 0.0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            frames, target, label = batch

            frames = frames.to(device)
            target = target.to(device)
            label = label.to(device)

            pred = model(frames)
            prev_frame = frames[:, -1]

            loss_dict = criterion(pred, target, prev_frame)

            total_loss += loss_dict["total_loss"].item()
            total_mse += loss_dict["mse_loss"].item()
            total_ssim += loss_dict["ssim_loss"].item()
            total_temp += loss_dict["temp_loss"].item()
            total_grad += loss_dict["grad_loss"].item()

            # ==========================================
            # ✅ PSNR + SSIM based anomaly score
            # ==========================================

            # PSNR (higher = normal)
            psnr = compute_psnr(pred, target)  # (B,)

            # SSIM (higher = normal)
            ssim = compute_ssim_score(loss_dict["ssim_loss"])  # (B,)

            # Normalize both
            #psnr_norm = normalize_tensor(psnr)
            #ssim_norm = normalize_tensor(ssim)

            # Convert to anomaly score
            # (low PSNR + low SSIM → high anomaly)
            anomaly_score = 0.99*(1 - psnr) + 0.01*(1 - ssim)

            all_scores.extend(anomaly_score.detach().cpu().numpy().tolist())
            all_labels.extend(label.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    avg_temp = total_temp / len(val_loader)
    avg_grad = total_grad / len(val_loader)

    auc = roc_auc_score(all_labels, all_scores)

    return avg_loss, avg_mse, avg_ssim, avg_temp, avg_grad, auc


# --------------------------------------------------
# Training Loop
# --------------------------------------------------
epochs = 200

for epoch in range(start_epoch, epochs):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_temp = 0.0
    total_grad = 0.0

    for frames, target in train_loader:
        frames = frames.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(frames)
        prev_frame = frames[:, -1]

        loss_dict = criterion(pred, target, prev_frame)
        loss = loss_dict["total_loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total_loss"].item()
        total_mse += loss_dict["mse_loss"].item()
        total_ssim += loss_dict["ssim_loss"].item()
        total_temp += loss_dict["temp_loss"].item()
        total_grad += loss_dict["grad_loss"].item()

    scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_mse = total_mse / len(train_loader)
    avg_train_ssim = total_ssim / len(train_loader)
    avg_train_temp = total_temp / len(train_loader)
    avg_train_grad = total_grad / len(train_loader)

    # ---------------- Validation ----------------
    val_loss, val_mse, val_ssim, val_temp, val_grad, val_auc = validate(
        model=model,
        val_loader=val_loader,
        device=device,
        criterion=criterion
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train MSE: {avg_train_mse:.4f} | "
        f"Train SSIM: {avg_train_ssim:.4f} | "
        f"Train Temp: {avg_train_temp:.4f} | "
        f"Train Grad: {avg_train_grad:.4f} || "
        f"Val Loss: {val_loss:.4f} | "
        f"Val MSE: {val_mse:.4f} | "
        f"Val SSIM: {val_ssim:.4f} | "
        f"Val Temp: {val_temp:.4f} | "
        f"Val Grad: {val_grad:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    # ---------------- Save last model ----------------
    torch.save(model.state_dict(), last_model_path)

    # ---------------- Save checkpoint ----------------
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
        },
        checkpoint_path
    )

    # ---------------- Save best model ----------------
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ New best model saved at epoch {epoch+1} | Best Val AUC: {best_auc:.4f}")

print("Training completed.")
print(f"Best validation AUC: {best_auc:.4f}")