import os
import torch
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()
DATALOADER_WORKERS = TRAIN_CONFIG["train_params"]["workers"]
CPU_THREADS_PER_PROC = max(1, CPU_COUNT // max(1, DATALOADER_WORKERS))

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)

torch.set_num_threads(CPU_THREADS_PER_PROC)

print(f"CPU count: {CPU_COUNT}")
print(f"DataLoader workers: {DATALOADER_WORKERS}")
print(f"CPU threads per process: {CPU_THREADS_PER_PROC}")
print(f"CUDA available: {torch.cuda.is_available()}")




# =========================================================
# PREFLIGHT CHECKS (GPU-AWARE, LOGGER-FRIENDLY)
# =========================================================

import os
import sys
import json
import yaml
import torch
import platform
import multiprocessing
from pathlib import Path

# ==========================
# USER CONFIG (from your setup)
# ==========================
DATA_YAML = PROJECT_ROOT / "config/data.yaml"
DATASET_ROOT = PROJECT_ROOT / "data/data_set/tomato-flower"
PROJECT_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "tomato_flower_yolo26_from_scratch"
MODEL_CFG = "yolo26n.yaml"

TRAIN_PARAMS = TRAIN_CONFIG["train_params"]

# ==========================
# CPU THREAD CONTROL (GPU TRAINING SAFE)
# ==========================
CPU_COUNT = multiprocessing.cpu_count()
WORKERS = TRAIN_PARAMS.get("workers", 8)

CPU_THREADS_PER_PROC = max(1, CPU_COUNT // max(1, WORKERS))

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS_PER_PROC)

torch.set_num_threads(CPU_THREADS_PER_PROC)

logger.info(
    f"\n=== CPU THREAD CONFIG ==="
    f"\n CPU count              : {CPU_COUNT}"
    f"\n DataLoader workers     : {WORKERS}"
    f"\n Threads per process    : {CPU_THREADS_PER_PROC}"
    f"\n torch num threads      : {torch.get_num_threads()}"
    f"\n"
)

# ==========================
# SYSTEM INFO
# ==========================
logger.info(
    f"\n=== SYSTEM INFO ==="
    f"\n Python executable : {sys.executable}"
    f"\n Python version    : {sys.version.splitlines()[0]}"
    f"\n Platform          : {platform.platform()}"
    f"\n Machine           : {platform.machine()}"
    f"\n Processor         : {platform.processor()}"
    f"\n CPU count         : {CPU_COUNT}"
    f"\n Working dir       : {Path.cwd()}"
    f"\n"
)

# ==========================
# TRAIN CONFIG SUMMARY
# ==========================
logger.info(
    f"\n=== TRAIN CONFIG SUMMARY ==="
    f"\n model_cfg   : {TRAIN_CONFIG['model_cfg']}"
    f"\n data_yaml   : {TRAIN_CONFIG['data_yaml']}"
    f"\n project_dir : {TRAIN_CONFIG['project_dir']}"
    f"\n run_name    : {TRAIN_CONFIG['run_name']}"
    f"\n epochs      : {TRAIN_PARAMS['epochs']}"
    f"\n imgsz       : {TRAIN_PARAMS['imgsz']}"
    f"\n batch       : {TRAIN_PARAMS['batch']}"
    f"\n device      : {TRAIN_PARAMS['device']}"
    f"\n workers     : {TRAIN_PARAMS['workers']}"
    f"\n amp         : {TRAIN_PARAMS['amp']}"
    f"\n optimizer   : {TRAIN_PARAMS['optimizer']}"
    f"\n pretrained  : {TRAIN_PARAMS['pretrained']}"
    f"\n"
)

# ==========================
# TORCH / CUDA INFO
# ==========================
cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

logger.info(
    f"\n=== TORCH INFO ==="
    f"\n torch version       : {torch.__version__}"
    f"\n CUDA available      : {cuda_available}"
    f"\n CUDA device count   : {torch.cuda.device_count() if cuda_available else 0}"
    f"\n MPS available       : {mps_available}"
    f"\n"
)

if cuda_available:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)

        logger.info(
            f"\nCUDA DEVICE {i}:"
            f"\n name              = {torch.cuda.get_device_name(i)}"
            f"\n capability        = {props.major}.{props.minor}"
            f"\n total_memory_gb   = {total_mem_gb:.2f}"
            f"\n multiprocessors   = {props.multi_processor_count}"
        )
else:
    logger.warning("CUDA is NOT available. Training will fall back to CPU.")

# ==========================
# CUDA SMOKE TEST
# ==========================
if cuda_available:
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        logger.info(f"CUDA smoke test PASSED (tensor device = {x.device})")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"CUDA detected but failed runtime test: {e}")

# ==========================
# ENV VARS
# ==========================
logger.info(
    f"\n=== IMPORTANT ENV VARS ==="
    f"\n OMP_NUM_THREADS        = {os.environ.get('OMP_NUM_THREADS')}"
    f"\n MKL_NUM_THREADS        = {os.environ.get('MKL_NUM_THREADS')}"
    f"\n NUMEXPR_NUM_THREADS    = {os.environ.get('NUMEXPR_NUM_THREADS')}"
    f"\n OPENBLAS_NUM_THREADS   = {os.environ.get('OPENBLAS_NUM_THREADS')}"
    f"\n CUDA_VISIBLE_DEVICES   = {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    f"\n ULTRALYTICS_CONFIG_DIR = {os.environ.get('ULTRALYTICS_CONFIG_DIR')}"
    f"\n"
)

if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    logger.info(
        f"CUDA_VISIBLE_DEVICES is set to {os.environ.get('CUDA_VISIBLE_DEVICES')} "
        "→ This may restrict visible GPUs."
    )

# ==========================
# DEVICE VALIDATION
# ==========================
requested_device = TRAIN_PARAMS["device"]

if requested_device != "cpu" and not cuda_available:
    raise RuntimeError(
        f"Config requests GPU (device={requested_device}) but CUDA is unavailable."
    )

if requested_device == -1:
    logger.info("device=-1 → auto-selecting most idle GPU.")

# ==========================
# HELPERS
# ==========================
def count_files(path, exts=None):
    if not path.exists():
        return 0
    files = [p for p in path.iterdir() if p.is_file()]
    if exts:
        exts = {e.lower() for e in exts}
        files = [p for p in files if p.suffix.lower() in exts]
    return len(files)

def print_path_status(label, path):
    path = Path(path)
    logger.info(
        f"\n{label}:"
        f"\n path    = {path}"
        f"\n exists  = {path.exists()}"
        f"\n is_dir  = {path.is_dir()}"
        f"\n is_file = {path.is_file()}"
    )

# ==========================
# PATH CHECKS
# ==========================
logger.info("\n=== PATH CHECKS ===")

print_path_status("DATA_YAML", DATA_YAML)
print_path_status("DATASET_ROOT", DATASET_ROOT)
print_path_status("PROJECT_DIR", PROJECT_DIR)

for split in ["train", "val", "test"]:
    split_dir = DATASET_ROOT / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    print_path_status(f"{split.upper()} DIR", split_dir)

    img_count = count_files(img_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    lbl_count = count_files(lbl_dir, {".txt"})

    logger.info(
        f"{split.upper()} COUNTS:"
        f"\n image files = {img_count}"
        f"\n label files = {lbl_count}"
    )

# ==========================
# YAML VALIDATION
# ==========================
logger.info("\n=== YAML CONTENTS ===")

with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

logger.info(json.dumps(data_cfg, indent=2))

logger.info("\n=== YAML PATH VALIDATION ===")

for key in ["train", "val", "test"]:
    p = Path(data_cfg[key])
    logger.info(
        f"{key}: {p}"
        f"\n exists = {p.exists()}"
        f"\n is_dir = {p.is_dir()}"
    )

logger.info("\n=== PREFLIGHT CHECK COMPLETE ===\n")
