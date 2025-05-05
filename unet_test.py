#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ————————————————————————————————————————————————
# reuse your classes from unet.py:
class NPYSegDataset(Dataset):
    """Load 2D .npy image/label pairs."""
    def __init__(self, img_dir, lbl_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        self.lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.npy")))
        assert len(self.img_paths) == len(self.lbl_paths), (
            f"Found {len(self.img_paths)} images but {len(self.lbl_paths)} labels"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)      # (H, W)
        lbl = np.load(self.lbl_paths[idx]).astype(np.int64)        # (H, W)
        img = np.expand_dims(img, 0)                               # (1, H, W)
        return torch.from_numpy(img), torch.from_numpy(lbl), os.path.basename(self.img_paths[idx])

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class TwoLayerUNet(nn.Module):
    def __init__(self, in_channels=1, base_features=32, n_classes=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features*2)
        self.up = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features*2, base_features)
        self.out = nn.Conv2d(base_features, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x5 = torch.cat([x4, x1], dim=1)
        return self.out(self.dec1(x5))
# ————————————————————————————————————————————————

def dice_coeff(pred, target, eps=1e-6):
    # assumes pred & target are torch tensors of shape (N,H,W) binary (0/1)
    intersection = (pred & target).float().sum((1,2))
    union = pred.float().sum((1,2)) + target.float().sum((1,2))
    return ((2 * intersection + eps) / (union + eps)).cpu().numpy()

def precision(pred, target, eps=1e-6):
    tp = (pred & target).float().sum((1,2))
    fp = (pred & (~target)).float().sum((1,2))
    return (tp / (tp + fp + eps)).cpu().numpy()

def recall(pred, target, eps=1e-6):
    tp = (pred & target).float().sum((1,2))
    fn = ((~pred) & target).float().sum((1,2))
    return (tp / (tp + fn + eps)).cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="../DATA/test_slices", required=False,
                   help="root folder with test_slices/images and /labels subdirs")
    p.add_argument("--model_path", default="checkpoints/two_layer_unet_epoch_13.pth",  required=False,
                   help="path to your trained two_layer_unet.pth")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=1)
    p.add_argument("--csv_out",      default="unet_test_metrics.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset & loader
    img_dir = os.path.join(args.test_dir, "images")
    lbl_dir = os.path.join(args.test_dir, "labels")
    ds = NPYSegDataset(img_dir, lbl_dir)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # model
    model = TwoLayerUNet(in_channels=1, base_features=32, n_classes=4)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    records = []
    with torch.no_grad():
        for imgs, lbls, names in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)              # (B,1,H,W)
            lbls = lbls.to(device)
            logits = model(imgs)                # (B,2,H,W)
            preds = torch.argmax(logits, dim=1) # (B,H,W)

            # for binary DSC, convert to bool masks
            pred_mask = preds.bool()
            true_mask = (lbls==1)

            # compute per-slice metrics
            d = dice_coeff(pred_mask, true_mask)
            p_ = precision(pred_mask, true_mask)
            r_ = recall(pred_mask, true_mask)

            for name, di, pi, ri in zip(names, d, p_, r_):
                records.append({
                    "slice": name,
                    "dice":    di,
                    "precision": pi,
                    "recall":    ri,
                })

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.csv_out, index=False)
    print(f"Wrote {len(df)} rows to {args.csv_out}")

if __name__=="__main__":
    main()