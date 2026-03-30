from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_LOCATION = Path("/workspace/flower-classifier/config/YOLO26_config.json")


def save_json_config(config: dict, path: str | Path = DEFAULT_LOCATION):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"DONE! Saved config to {path}")


def load_json_config(path: str | Path = DEFAULT_LOCATION) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        config = json.load(f)
    logger.info(f"DONE! Loaded config from {path}")
    return config


def make_YOLO26_config(file_path = '/workspace/flower-classifier/config/YOLO26_config.json'):
    """
    Creates a config file at the specified file path with all the YOLO26 config fields with default values.
    """

    TRAIN_CONFIG = {
        "model_cfg": "yolo26n.yaml",
        "data_yaml": "/workspace/flower-classifier/config/data.yaml",
        "project_dir": "/workspace/flower-classifier/runs",
        "run_name": "tomato_flower_yolo26_from_scratch",
        "train_params": {
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "device": "cpu",
            "workers": 4,
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
            "amp": False,
            "plots": True,
            "save": True,
            "save_period": -1,
            "exist_ok": True,
            "pretrained": False,
            "verbose": True
        }
    }

    try:
        with open(file_path, 'w') as json_file:
            json.dump(TRAIN_CONFIG, json_file, indent=4)
        logger.info(f"Data successfully saved to {file_path}")
    except IOError as e:
        logger.info(f"Error saving file: {e}")