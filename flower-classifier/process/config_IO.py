from pathlib import Path
import json
import logging
import re

logger = logging.getLogger(f"flower_classifier.{__name__}")

path = Path(__file__).resolve().parent
while path.name != "flower-classifier" and path != path.parent:
    path = path.parent
logger.info(f"Path to project parent directory: '{path}'")
PROJECT_ROOT = path
DEFAULT_LOCATION = PROJECT_ROOT / "config/YOLO26_config.json"


def save_json_config(config: dict, file_path: str | Path = DEFAULT_LOCATION):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"DONE! Saved config to {file_path}")


def load_json_config(file_path: str | Path = DEFAULT_LOCATION) -> dict:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        config = json.load(f)
    logger.info(f"DONE! Loaded config from {file_path}")
    logger.info(f"Loaded config set: \n{json.dumps(config, indent=4)}")
    return config


def make_YOLO26_config(file_path: str | Path = DEFAULT_LOCATION):
    """
    Creates a config file at the specified file path with all the YOLO26 config fields with default values.
    """

    TRAIN_CONFIG = {
        "model_cfg": "yolo26n.yaml",
        "data_yaml": str(PROJECT_ROOT / "config/data.yaml"),
        "project_dir": str(PROJECT_ROOT / "runs"),
        "run_name": "exp_",
        "train_params": {
            "epochs": 100,
            "imgsz": 640,
            "batch": -1,
            "device": -1,
            "workers": 8,
            "patience": 30,
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "close_mosaic": 10,
            "single_cls": False,
            "cos_lr": False,
            "amp": True,
            "plots": True,
            "save": True,
            "save_period": -1,
            "exist_ok": False,
            "pretrained": False,
            "verbose": True
        }
    }

    logger.info(f"Creating default config set at location '{file_path}'")
    try:
        file_path = Path(file_path)
        logger.info(f"Default config set: \n{json.dumps(TRAIN_CONFIG, indent=4)}")
        with open(file_path, 'w') as json_file:
            json.dump(TRAIN_CONFIG, json_file, indent=4)
        logger.info(f"Data successfully saved to {file_path}")
    except IOError as e:
        logger.info(f"Error saving file: {e}")


def get_highest_numbered_run(parent_path):
    """
    Helper function which gets the highest-number folder in the runs subdirectory and returns it.
    """
    path = Path(parent_path)
    
    # Get all subdirectories in the location
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    
    # Helper to extract the number from the end of the name
    def extract_suffix_number(folder):
        # Finds one or more digits (\d+) at the end ($) of the folder name
        match = re.search(r'(\d+)$', folder.name)
        return int(match.group(1)) if match else 0

    # Find the folder with the maximum numeric suffix
    if not subdirs:
        return 0
        
    return extract_suffix_number(max(subdirs, key=extract_suffix_number))