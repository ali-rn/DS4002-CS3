#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -- your dataset class, just like in test.py --
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
        return torch.from_numpy(img), torch.from_numpy(lbl)

# -- model definition --
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
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features*2)
        # Decoder
        self.up = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features*2, base_features)
        # Output
        self.out = nn.Conv2d(base_features, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        # assume x4 and x1 spatial dims match
        x5 = torch.cat([x4, x1], dim=1)
        return self.out(self.dec1(x5))

# -- training & validation loops --
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    pbar = tqdm(loader, desc="Train" if train else " Val ", leave=False)
    with torch.set_grad_enabled(train):
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss = criterion(logits, lbls)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=running_loss/((pbar.n+1)*imgs.size(0)))
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="../DATA/train_slices", help="folder with images/ and labels/")
    parser.add_argument("--val_dir",   default="../DATA/val_slices",   help="folder with images/ and labels/")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch_size",type=int,   default=8)
    parser.add_argument("--in_ch",     type=int,   default=1)
    parser.add_argument("--classes",   type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    # datasets & loaders
    train_ds = NPYSegDataset(os.path.join(args.train_dir, "images"),
                             os.path.join(args.train_dir, "labels"))
    val_ds   = NPYSegDataset(os.path.join(args.val_dir,   "images"),
                             os.path.join(args.val_dir,   "labels"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=1)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=1)

    # model, loss, optimizer
    model = TwoLayerUNet(in_channels=args.in_ch, base_features=32, n_classes=args.classes)
    model = model.to(device)
    # create checkpoint directory
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    # perform validation only every 20% of total epochs (5 times)
    val_interval = max(1, args.epochs // 5)
    for epoch in range(1, args.epochs+1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        if epoch % val_interval == 0:
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            print(f"Epoch {epoch:02d}/{args.epochs} → train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            # save checkpoint after this epoch
            ckpt_path = os.path.join(ckpt_dir, f"two_layer_unet_epoch_{epoch:02d}.pth")
            torch.save(model.state_dict(), ckpt_path)
        else:
            print(f"Epoch {epoch:02d}/{args.epochs} → train_loss: {train_loss:.4f}")
            # save checkpoint after this epoch
            ckpt_path = os.path.join(ckpt_dir, f"two_layer_unet_epoch_{epoch:02d}.pth")
            torch.save(model.state_dict(), ckpt_path)

    # save
    torch.save(model.state_dict(), "two_layer_unet.pth")
    print("Model checkpoint saved to two_layer_unet.pth")

if __name__ == "__main__":
    main()