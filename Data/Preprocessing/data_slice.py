# Script to slice 3D NIfTI volumes into 2D NumPy arrays per split

import os
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm
import torch  # Add this at the top with other imports
np.float = float
import nibabel as nib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_file(paths, dst_img_dir, dst_lbl_dir, axis):
    img_path, lbl_path = paths
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    img_arr = img_nii.get_fdata().astype(np.float32)
    lbl_arr = lbl_nii.get_fdata().astype(np.int16)
    
    # print(f"NIfTI shape: {img_arr.shape}")

    basename = os.path.splitext(os.path.basename(img_path))[0]
    if basename.endswith('.nii'):
        basename = basename[:-4]

    # print(f"Processing '{basename}': image shape {img_arr.shape}, label shape {lbl_arr.shape}")

    img_tensor = torch.from_numpy(img_arr).to(DEVICE)
    lbl_tensor = torch.from_numpy(lbl_arr).to(DEVICE)

    depth = img_arr.shape[axis]
    for idx in range(depth):
        # extract 2D slice along axis
        img_slice = torch.index_select(img_tensor, axis, torch.tensor([idx], device=DEVICE)).squeeze().cpu().numpy()
        lbl_slice = torch.index_select(lbl_tensor, axis, torch.tensor([idx], device=DEVICE)).squeeze().cpu().numpy()

        # save reshaped slices
        img_filename = f"{basename}_slice{idx:03d}.npy"
        lbl_filename = f"{basename}_slice{idx:03d}.npy"
        np.save(os.path.join(dst_img_dir, img_filename), img_slice)
        np.save(os.path.join(dst_lbl_dir, lbl_filename), lbl_slice)

def slice_nifti_to_numpy(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, axis=0):
    """
    Slices 3D NIfTI images and labels into 2D NumPy arrays along the specified axis.

    Args:
        src_img_dir (str): Directory containing source image .nii.gz files.
        src_lbl_dir (str): Directory containing source label .nii.gz files.
        dst_img_dir (str): Directory to save sliced image .npy files.
        dst_lbl_dir (str): Directory to save sliced label .npy files.
        axis (int): Axis along which to slice (0, 1, or 2). Default is 0 (axial).
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    images = sorted(glob(os.path.join(src_img_dir, "*.nii.gz")))
    labels = sorted(glob(os.path.join(src_lbl_dir, "*.nii.gz")))

    assert len(images) == len(labels), f"Found {len(images)} images and {len(labels)} labels"

    # Serial processing with progress bar
    for paths in tqdm(zip(images, labels), total=len(images), desc=f"Slicing volumes in {src_img_dir}"):
        process_file(paths, dst_img_dir=dst_img_dir, dst_lbl_dir=dst_lbl_dir, axis=axis)

    print(f"Sliced {len(images)} volumes along axis {axis}.")


def main():
    # Splits to process: train, val, test
    for split in ["train", "test", "val"]:
        src_img = os.path.join("../DATA", split, "images")
        src_lbl = os.path.join("../DATA", split, "labels")
        dst_img = os.path.join("../DATA", f"{split}_slices", "images")
        dst_lbl = os.path.join("../DATA", f"{split}_slices", "labels")
        print(f"Processing split: {split}")
        slice_nifti_to_numpy(src_img, src_lbl, dst_img, dst_lbl)


if __name__ == "__main__":
    main()
