## INGESTION
## SPLIT VALID INTO VAL AND TEST

from pathlib import Path
import shutil
import random
import yaml
import os
import logging

logger = logging.getLogger(__name__)

# # ==========================
# # CONFIG
# # ==========================
# DATASET_ROOT = Path("/workspace/data/data_set/tomato-flower")
# SOURCE_DIR = DATASET_ROOT / "valid"
# VAL_RATIO = 0.5
# RANDOM_SEED = 42

def ingest(
        dataset_root=Path("/workspace/data/data_set/tomato-flower"), 
        source_dir=Path("/workspace/data/data_set/tomato-flower/valid"), 
        val_ratio=0.5, 
        random_seed=42):
    """
    Extracts the zipped dataset and prepares the project directory.
    """

    # ==========================
    # PATHS
    # ==========================
    source_images = source_dir / "images"
    source_labels = source_dir / "labels"

    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    val_images = val_dir / "images"
    val_labels = val_dir / "labels"
    test_images = test_dir / "images"
    test_labels = test_dir / "labels"

    # Create output directories
    for d in [val_images, val_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # ==========================
    # COLLECT IMAGE FILES
    # ==========================
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in source_images.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    if not image_files:
        raise ValueError(f"No image files found in {source_images}")

    # ==========================
    # SHUFFLE + SPLIT
    # ==========================
    random.seed(random_seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * val_ratio)
    val_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    logger.info(f"Total source images: {len(image_files)}")
    logger.info(f"Validation images:   {len(val_files)}")
    logger.info(f"Test images:         {len(test_files)}")

    # ==========================
    # MOVE IMAGE + LABEL PAIR
    # ==========================
    def move_pair(img_path: Path, src_label_dir: Path, dest_img_dir: Path, dest_label_dir: Path):
        label_path = src_label_dir / f"{img_path.stem}.txt"

        # Move image
        shutil.copy2(str(img_path), str(dest_img_dir / img_path.name))

        # Move matching label
        if label_path.exists():
            shutil.copy2(str(label_path), str(dest_label_dir / label_path.name))
        else:
            logger.warning(f"Missing label for image: {img_path.name}")

    # Move val split
    for img in val_files:
        move_pair(img, source_labels, val_images, val_labels)

    # Move test split
    for img in test_files:
        move_pair(img, source_labels, test_images, test_labels)

    logger.info("DONE! Split complete.")



    ## VERIFY INGESTION
    ## COUNT AMOUNT OF IMAGES IN EACH FOLDER
    for split in ["train", "val", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"

        img_count = len([p for p in img_dir.iterdir() if p.is_file()]) if img_dir.exists() else 0
        lbl_count = len([p for p in lbl_dir.iterdir() if p.is_file()]) if lbl_dir.exists() else 0

        logger.info(f" \
            \n {split.upper()}: \
            \n   images = {img_count} \
            \n   labels = {lbl_count} \
            \n ")


    ## DATA YAML UPDATER
    ## CREATES THE NEW DATA CONFIG FILE FOR THE NEW FOLDERS
    data_yaml_path = Path("/workspace/flower-classifier/config/data.yaml")
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    data_config = {
        "train": "/workspace/data/data_set/tomato-flower/train/images",
        "val": "/workspace/data/data_set/tomato-flower/val/images",
        "test": "/workspace/data/data_set/tomato-flower/test/images",
        "nc": 3,
        "names": ["Bud", "Anthesis", "Post-Anthesis"],
    }

    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_config, f, sort_keys=False)

    logger.info(f"DONE! Wrote data.yaml to: {data_yaml_path}")
    logger.info(data_yaml_path.read_text())



    ## CREATE RUNS DIRECTORY
    ## If the directory where runout files will be stored does not yet exist, make it.
    directory_path = '/workspace/flower-classifier/runs'
    os.makedirs(directory_path, exist_ok=True)
