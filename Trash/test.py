#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Precision, Recall
from monai.metrics import DiceMetric, SurfaceDiceMetric
import segmentation_models_pytorch as smp
import pandas as pd
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

class NPYSegDataset(Dataset):
    """Load 2D .npy image/label pairs and return paths for CSV logging."""
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
        img_t = torch.from_numpy(img)
        lbl_t = torch.from_numpy(lbl)
        path = os.path.basename(self.img_paths[idx])
        return img_t, lbl_t, path


def main():
    parser = argparse.ArgumentParser(description="Test segmentation model and save metrics CSV")
    parser.add_argument("--model", required=False, default="deeplabv3p_mobilenet2d.pth", help="path to .pth checkpoint")
    parser.add_argument("--test_dir", default="../DATA/test_slices",
                        help="root with subfolders images/ and labels/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--tol", type=float, default=2.0,
                        help="surface DSC tolerance in pixels")
    parser.add_argument("--csv", default="metrics.csv",
                        help="output CSV file path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=1,
        classes=args.classes,
    ).to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Data
    img_dir = os.path.join(args.test_dir, "images")
    lbl_dir = os.path.join(args.test_dir, "labels")
    ds = NPYSegDataset(img_dir, lbl_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Collect predictions per subject
    subj_preds = defaultdict(list)
    subj_lbls  = defaultdict(list)

    with torch.no_grad():
        for imgs, lbls, paths in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            logits = model(imgs)
            preds  = torch.argmax(logits, dim=1)
            for i, path in enumerate(paths):
                subj = path.split("_slice_")[0]
                subj_preds[subj].append(preds[i].cpu())
                subj_lbls[subj].append(lbls[i].cpu())

    # Compute per‑subject metrics
    records = []
    for subj in sorted(subj_preds):
        # stack to volume (D, H, W)
        pred_vol = torch.stack(subj_preds[subj], dim=0)
        lbl_vol  = torch.stack(subj_lbls[subj],  dim=0)

        # one-hot for Dice/Surface‑DSC
        pred_oh = torch.nn.functional.one_hot(pred_vol, num_classes=args.classes)\
                        .permute(3,0,1,2).unsqueeze(0).float().to(device)
        lbl_oh  = torch.nn.functional.one_hot(lbl_vol,  num_classes=args.classes)\
                        .permute(3,0,1,2).unsqueeze(0).float().to(device)

        dsc  = dice_metric(y_pred=pred_oh, y=lbl_oh).item()
        sds  = surface_metric(y_pred=pred_oh, y=lbl_oh).item()

        # flatten for precision/recall
        pf = pred_vol.flatten().unsqueeze(0).to(device)
        lf = lbl_vol.flatten().unsqueeze(0).to(device)
        precision_metric.reset(); recall_metric.reset()
        prec = precision_metric(pf, lf).item()
        rec  = recall_metric(pf, lf).item()

        records.append({
            "subject": subj,
            "dsc": dsc,
            "surface_dsc": sds,
            "precision": prec,
            "recall": rec
        })

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(args.csv, index=False)
    print(f"Wrote metrics for {len(df)} subjects to {args.csv}")


if __name__ == "__main__":
    main()
