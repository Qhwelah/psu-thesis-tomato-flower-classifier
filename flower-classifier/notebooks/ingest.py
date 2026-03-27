import os
import shutil
import random
from pathlib import Path

# ==========================
# CONFIG
# ==========================
DATASET_ROOT = Path("/workspace/data/data_set/tomato-flower")
VALID_DIR = DATASET_ROOT / "valid"
VAL_RATIO = 0.5  # 50% val, 50% test

# ==========================
# PATHS
# ==========================
valid_images = VALID_DIR / "images"
valid_labels = VALID_DIR / "labels"

val_dir = DATASET_ROOT / "val"
test_dir = DATASET_ROOT / "test"

val_images = val_dir / "images"
val_labels = val_dir / "labels"

test_images = test_dir / "images"
test_labels = test_dir / "labels"

# Create output directories
for d in [val_images, val_labels, test_images, test_labels]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================
# GET FILE LIST
# ==========================
image_files = sorted(list(valid_images.glob("*.jpg")))

# Shuffle for randomness
random.shuffle(image_files)

# Split
split_idx = int(len(image_files) * VAL_RATIO)
val_files = image_files[:split_idx]
test_files = image_files[split_idx:]

print(f"Total images: {len(image_files)}")
print(f"Val: {len(val_files)} | Test: {len(test_files)}")

# ==========================
# MOVE FUNCTION
# ==========================
def move_pair(img_path, src_label_dir, dest_img_dir, dest_label_dir):
    label_path = src_label_dir / (img_path.stem + ".txt")

    # Move image
    shutil.copy2(str(img_path), dest_img_dir / img_path.name)

    # Move label if exists
    if label_path.exists():
        shutil.copy2(str(label_path), dest_label_dir / label_path.name)
    else:
        print(f"WARNING! Missing label for {img_path.name}")

# ==========================
# MOVE FILES
# ==========================
# Move validation split
for img in val_files:
    move_pair(img, valid_labels, val_images, val_labels)

# Move test split
for img in test_files:
    move_pair(img, valid_labels, test_images, test_labels)

print("DONE! Split complete.")