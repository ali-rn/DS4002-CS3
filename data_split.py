import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

def main():
    # Define source directories
    images_dir = os.path.join("../DATA", "images")
    labels_dir = os.path.join("../DATA", "labels")

    # Collect and sort image/label filepaths
    images = sorted(glob(os.path.join(images_dir, "*.nii.gz")))
    labels = sorted(glob(os.path.join(labels_dir, "*.nii.gz")))

    # Ensure they match one-to-one
    assert len(images) == len(labels), f"Found {len(images)} images and {len(labels)} labels"

    # Pair up
    data = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

    # Split off 20% for test
    train_val, test_data = train_test_split(data, test_size=0.2, random_state=42)
    # From remaining 80%, split off 20% for validation (16% total)
    train_data, val_data = train_test_split(train_val, test_size=0.2, random_state=42)

    def copy_subset(subset, subset_name):
        out_img = os.path.join("../DATA", subset_name, "images")
        out_lbl = os.path.join("../DATA", subset_name, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        for item in subset:
            shutil.copy(item["image"], os.path.join(out_img, os.path.basename(item["image"])))
            shutil.copy(item["label"], os.path.join(out_lbl, os.path.basename(item["label"])))
        print(f"Copied {len(subset)} cases to DATA/{subset_name}/")

    # Copy each subset into its own folder
    copy_subset(test_data, "test")
    copy_subset(val_data, "val")
    copy_subset(train_data, "train")

if __name__ == "__main__":
    main()
