import os
import shutil

def main():
    # Source KITs23 dataset root
    src_root = os.path.join("../../kits23", "dataset")
    
    # Destination directories
    dst_images = os.path.join("../DATA", "images")
    dst_labels = os.path.join("../DATA", "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    
    # Loop over each case directory
    selected_cases = sorted(os.listdir(src_root))[:110]
    for case in selected_cases:
        case_dir = os.path.join(src_root, case)
        if not os.path.isdir(case_dir):
            continue
        # Paths to imaging and segmentation files
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        # Copy if both exist
        if os.path.exists(img_path) and os.path.exists(seg_path):
            shutil.copy(img_path, os.path.join(dst_images, f"{case}.nii.gz"))
            shutil.copy(seg_path, os.path.join(dst_labels, f"{case}.nii.gz"))
        else:
            print(f"Warning: missing files for {case}, skipping")
    
    print(f"Reorganized {len(os.listdir(dst_images))} images and {len(os.listdir(dst_labels))} labels.")

if __name__ == "__main__":
    main()
