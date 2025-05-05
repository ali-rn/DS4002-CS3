#!/usr/bin/env python3
import os
import glob
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import segmentation_models_pytorch as smp


class NPYSegDataset(Dataset):
    """Simple dataset for loading 2D .npy image/label pairs."""
    def __init__(self, img_dir, lbl_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        self.lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.npy")))
        assert len(self.img_paths) == len(self.lbl_paths), (
            f"Found {len(self.img_paths)} images but {len(self.lbl_paths)} labels"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # load numpy arrays
        img = np.load(self.img_paths[idx]).astype(np.float32)
        lbl = np.load(self.lbl_paths[idx]).astype(np.int64)  # integer class labels

        # add channel dimension (1, H, W)
        img = np.expand_dims(img, axis=0)

        # to tensor
        img_t = torch.from_numpy(img)
        lbl_t = torch.from_numpy(lbl)
        return img_t, lbl_t


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, lbls in pbar:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss = criterion(preds, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val", leave=False)
        for imgs, lbls in pbar:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            preds = model(imgs)
            loss = criterion(preds, lbls)
            total_loss += loss.item()
            pbar.set_postfix(val_loss=f"{total_loss/(pbar.n+1):.4f}")

    return total_loss / len(loader)


def main():
    # --- Configuration ---
    DATA_ROOT = "../DATA"
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_slices", "images")
    TRAIN_LBL_DIR = os.path.join(DATA_ROOT, "train_slices", "labels")
    VAL_IMG_DIR   = os.path.join(DATA_ROOT, "val_slices",   "images")
    VAL_LBL_DIR   = os.path.join(DATA_ROOT, "val_slices",   "labels")

    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LR = 2e-4
    MAX_EPOCHS = 30

    # --- Device ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- Datasets & Loaders ---
    train_ds = NPYSegDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    val_ds   = NPYSegDataset(VAL_IMG_DIR,   VAL_LBL_DIR)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # --- Model, Loss, Optimizer ---
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=1,
        classes=4,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    # --- Training Loop ---
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss   = validate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

    # --- Save final model ---
    torch.save(model.state_dict(), "deeplabv3p_mobilenet2d.pth")
    print("Training complete. Model saved to deeplabv3p_mobilenet2d.pth")


if __name__ == "__main__":
    main()
