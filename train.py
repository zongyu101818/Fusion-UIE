import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import lpips
from collections import namedtuple

from models import EnhancedCC_Module
from utils.dataset import Dataset_Load, ToTensor
from utils.metrics import getUIQM, getSSIM, getPSNR
from utils.helpers import get_lr
from brisque import BRISQUE


# =========================
# Utilities
# =========================
def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_batch(batch):
    """Normalize using imagenet mean and std"""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def compute_lpips_metric(pred, target, lpips_loss_fn):
    """Calculate LPIPS perceptual distance. pred/target: (1,3,H,W) in [0,1]."""
    pred_norm = pred * 2 - 1
    target_norm = target * 2 - 1
    return lpips_loss_fn(pred_norm, target_norm).mean().item()


def find_resume_checkpoint(checkpoints_dir: str) -> str | None:
    """
    Recommended resume behavior:
    1) use netG_last.pt if exists (most recent)
    2) else fallback to netG_best.pt
    3) else None
    """
    last_path = os.path.join(checkpoints_dir, "netG_last.pt")
    best_path = os.path.join(checkpoints_dir, "netG_best.pt")
    if os.path.exists(last_path):
        return last_path
    if os.path.exists(best_path):
        return best_path
    return None


# =========================
# VGG16 for perceptual loss
# =========================
class Vgg16(torch.nn.Module):
    """VGG16 model for perceptual loss calculation"""

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)


# =========================
# Main training
# =========================
def train(config, data_root):
    # -------------------------
    # Checkpoint config (NEW)
    # -------------------------
    checkpoints_dir = config["training"]["checkpoints_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)

    # How often to keep an archival snapshot (optional).
    # Set to 0 to disable.
    SAVE_EVERY_N_EPOCHS = 0  # e.g. 20 if you want netG_20.pt, netG_40.pt... kept

    # Choose "best" criterion
    # Recommended: best by val_psnr
    BEST_MODE = "psnr"  # "psnr" or "loss"
    best_score = -1e18 if BEST_MODE == "psnr" else 1e18

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Model + losses
    # -------------------------
    netG = EnhancedCC_Module().to(device)

    mse_loss = nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    brisque_obj = BRISQUE(url=False)

    optim_g = optim.Adam(
        netG.parameters(),
        lr=config["training"]["learning_rate_g"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        weight_decay=config["training"]["wd_g"],
    )

    lambda_mse = float(config["training"]["lambda_mse"])
    lambda_vgg = float(config["training"]["lambda_vgg"])

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = Dataset_Load(
        data_root=data_root, dataset_name="UIEB", transform=ToTensor(), train=True
    )
    val_dataset = Dataset_Load(
        data_root=data_root, dataset_name="UIEB", transform=ToTensor(), train=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    train_batches = len(train_dataloader)
    val_batches = len(val_dataloader)

    # -------------------------
    # Resume (NEW)
    # -------------------------
    resume_path = find_resume_checkpoint(checkpoints_dir)
    print(f"Loading model for generator (resume): {resume_path}")

    if resume_path is None:
        start_epoch = 1
        print("No checkpoints found for netG! Starting training from scratch")
    else:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = int(ckpt["epoch"]) + 1
        netG.load_state_dict(ckpt["model_state_dict"])
        optim_g.load_state_dict(ckpt["optimizer_state_dict"])

        # restore best score if we loaded best/last that contains it
        if BEST_MODE == "psnr" and ("val_psnr" in ckpt):
            best_score = float(ckpt["val_psnr"])
        if BEST_MODE == "loss" and ("val_total_loss" in ckpt):
            best_score = float(ckpt["val_total_loss"])

        print(f"Restoring model from checkpoint epoch {start_epoch}")

    # -------------------------
    # Training loop
    # -------------------------
    end_epoch = int(config["training"]["end_epoch"])

    for epoch in range(start_epoch, end_epoch + 1):
        # ===== Train =====
        netG.train()
        total_train_mse_loss = 0.0
        total_train_vgg_loss = 0.0
        total_train_G_loss = 0.0

        for i_batch, sample_batched in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
        ):
            hazy_batch = sample_batched["hazy"].to(device, non_blocking=True)
            clean_batch = sample_batched["clean"].to(device, non_blocking=True)

            optim_g.zero_grad()

            pred_batch = netG(hazy_batch)

            # MSE
            batch_mse = lambda_mse * mse_loss(pred_batch, clean_batch)
            batch_mse.backward(retain_graph=True)

            # VGG perceptual
            clean_vgg_feats = vgg(normalize_batch(clean_batch))
            pred_vgg_feats = vgg(normalize_batch(pred_batch))
            batch_vgg = lambda_vgg * mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2)
            batch_vgg.backward()

            optim_g.step()

            batch_mse_val = float(batch_mse.item())
            batch_vgg_val = float(batch_vgg.item())
            batch_G_val = batch_mse_val + batch_vgg_val

            total_train_mse_loss += batch_mse_val
            total_train_vgg_loss += batch_vgg_val
            total_train_G_loss += batch_G_val

            if (i_batch + 1) % 50 == 0:
                print(
                    f"\rEpoch: {epoch} | Train: ({i_batch+1}/{train_batches}) "
                    f"| g_mse: {batch_mse_val:.6f} | g_vgg: {batch_vgg_val:.6f}"
                )

        avg_train_mse_loss = total_train_mse_loss / max(train_batches, 1)
        avg_train_vgg_loss = total_train_vgg_loss / max(train_batches, 1)
        avg_train_G_loss = total_train_G_loss / max(train_batches, 1)

        # ===== Val =====
        netG.eval()
        total_val_mse_loss = 0.0
        total_val_vgg_loss = 0.0
        total_val_G_loss = 0.0

        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_uiqm = 0.0
        total_brisque = 0.0

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(
                tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]")
            ):
                hazy_batch = sample_batched["hazy"].to(device, non_blocking=True)
                clean_batch = sample_batched["clean"].to(device, non_blocking=True)

                pred_batch = netG(hazy_batch)

                batch_mse_loss = float((lambda_mse * mse_loss(pred_batch, clean_batch)).item())

                clean_vgg_feats = vgg(normalize_batch(clean_batch))
                pred_vgg_feats = vgg(normalize_batch(pred_batch))
                batch_vgg_loss = float(
                    (lambda_vgg * mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2)).item()
                )

                total_val_mse_loss += batch_mse_loss
                total_val_vgg_loss += batch_vgg_loss
                total_val_G_loss += (batch_mse_loss + batch_vgg_loss)

                # Metrics per image
                pred_np = pred_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)   # B,H,W,C float (likely not clamped)
                clean_np = clean_batch.detach().cpu().numpy().transpose(0, 2, 3, 1) # B,H,W,C float

                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_lp = 0.0
                batch_u = 0.0
                batch_b = 0.0

                bs = pred_np.shape[0]
                for j in range(bs):
                    # IMPORTANT: clamp to [0,1] before converting metrics
                    pred01 = np.clip(pred_np[j], 0.0, 1.0)
                    clean01 = np.clip(clean_np[j], 0.0, 1.0)

                    pred_255 = (pred01 * 255.0).clip(0, 255).astype(np.float64)
                    clean_255 = (clean01 * 255.0).clip(0, 255).astype(np.float64)

                    batch_psnr += getPSNR(pred_255, clean_255)
                    batch_ssim += getSSIM(pred_255, clean_255)

                    img1 = torch.from_numpy(pred01.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    img2 = torch.from_numpy(clean01.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    batch_lp += compute_lpips_metric(img1, img2, lpips_loss_fn)

                    pred_uint8 = pred_255.astype(np.uint8)
                    batch_u += getUIQM(pred_uint8)
                    batch_b += brisque_obj.score(img=pred_uint8)

                batch_psnr /= max(bs, 1)
                batch_ssim /= max(bs, 1)
                batch_lp /= max(bs, 1)
                batch_u /= max(bs, 1)
                batch_b /= max(bs, 1)

                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_lpips += batch_lp
                total_uiqm += batch_u
                total_brisque += batch_b

                if (i_batch + 1) % 10 == 0:
                    print(
                        f"\rEpoch: {epoch} | Validation: ({i_batch+1}/{val_batches}) | "
                        f"g_mse: {batch_mse_loss:.6f} | g_vgg: {batch_vgg_loss:.6f} | "
                        f"psnr: {batch_psnr:.4f} | ssim: {batch_ssim:.4f} | "
                        f"lpips: {batch_lp:.4f} | uiqm: {batch_u:.4f}"
                    )

        avg_val_mse_loss = total_val_mse_loss / max(val_batches, 1)
        avg_val_vgg_loss = total_val_vgg_loss / max(val_batches, 1)
        avg_val_G_loss = total_val_G_loss / max(val_batches, 1)

        avg_val_psnr = total_psnr / max(val_batches, 1)
        avg_val_ssim = total_ssim / max(val_batches, 1)
        avg_val_lpips = total_lpips / max(val_batches, 1)
        avg_val_uiqm = total_uiqm / max(val_batches, 1)
        avg_val_brisque = total_brisque / max(val_batches, 1)

        print(f"\nEpoch {epoch} Summary:")
        print(
            "Training   - lr: %.6f, mse: %.6f, vgg: %.6f, total: %.6f"
            % (get_lr(optim_g), avg_train_mse_loss, avg_train_vgg_loss, avg_train_G_loss)
        )
        print(
            "Validation - mse: %.6f, vgg: %.6f, total: %.6f"
            % (avg_val_mse_loss, avg_val_vgg_loss, avg_val_G_loss)
        )
        print("Image Quality Metrics:")
        print("  - PSNR: %.4f" % avg_val_psnr)
        print("  - SSIM: %.4f" % avg_val_ssim)
        print("  - LPIPS: %.4f (lower is better)" % avg_val_lpips)
        print("  - UIQM: %.4f" % avg_val_uiqm)
        print("  - BRISQUE: %.4f (lower is better)" % avg_val_brisque)

        # =========================
        # Save checkpoints (NEW, recommended)
        # =========================
        ckpt = {
            "epoch": epoch,
            "model_state_dict": netG.state_dict(),
            "optimizer_state_dict": optim_g.state_dict(),
            "train_mse_loss": avg_train_mse_loss,
            "train_vgg_loss": avg_train_vgg_loss,
            "train_total_loss": avg_train_G_loss,
            "val_mse_loss": avg_val_mse_loss,
            "val_vgg_loss": avg_val_vgg_loss,
            "val_total_loss": avg_val_G_loss,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "val_lpips": avg_val_lpips,
            "val_uiqm": avg_val_uiqm,
            "val_brisque": avg_val_brisque,
        }

        # 1) Always save LAST (overwrite): only one file
        last_path = os.path.join(checkpoints_dir, "netG_last.pt")
        torch.save(ckpt, last_path)

        # 2) Save BEST (overwrite only when improved): only one file
        improved = False
        if BEST_MODE == "psnr":
            score = avg_val_psnr
            if score > best_score:
                best_score = score
                improved = True
        else:
            score = avg_val_G_loss
            if score < best_score:
                best_score = score
                improved = True

        if improved:
            best_path = os.path.join(checkpoints_dir, "netG_best.pt")
            torch.save(ckpt, best_path)
            print(f"[Checkpoint] New BEST ({BEST_MODE}={best_score:.4f}) -> saved netG_best.pt")

        # 3) Optional archival snapshots every N epochs
        if SAVE_EVERY_N_EPOCHS and (epoch % SAVE_EVERY_N_EPOCHS == 0):
            snap_path = os.path.join(checkpoints_dir, f"netG_{epoch}.pt")
            torch.save(ckpt, snap_path)
            print(f"[Checkpoint] Saved snapshot: {snap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FUSION")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/path/to/data",
        help="Path to dataset root directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.data_root)