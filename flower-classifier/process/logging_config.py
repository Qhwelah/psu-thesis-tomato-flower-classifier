import logging
import os, sys
from pathlib import Path
from ultralytics.utils import LOGGER as YOLO_LOGGER

LOG_FILE_ROOT = Path(__file__).resolve().parent
BASE_LOGGER_NAME = "flower_classifier"

def setup_logging(log_file: str|None = None) -> logging.Logger:

    # Define a default log name if not specified by caller.
    if not log_file:
        log_file = str(LOG_FILE_ROOT / "flower_classifier.log")

    log_path = Path(log_file)
    #os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Main app logger
    logger = logging.getLogger(BASE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Prevent duplicate handlers in Jupyter if re-running cells
    logger.handlers.clear()

    # File handler: overwrite each run
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Attach SAME file handler to Ultralytics logger
    # Avoid duplicate attachment if this function is rerun
    if file_handler not in YOLO_LOGGER.handlers:
        YOLO_LOGGER.addHandler(file_handler)

    logger.info("Writing log to: '%s'", log_file)

    return logger