import torch
import torchvision
import ultralytics
import cv2
import yaml
import captum
import sys, platform
import logging

logger = logging.getLogger(__name__)


def print_system_info():
    """
    Finds and prints out a report of all current system info.
    """
    
    logger.info("torch:", torch.__version__)
    logger.info("torchvision:", torchvision.__version__)
    logger.info("ultralytics:", ultralytics.__version__)
    logger.info("cv2:", cv2.__version__)
    logger.info("captum:", captum.__version__)
    logger.info("CUDA available:", torch.cuda.is_available())
    logger.info("Python:", sys.version)
    logger.info("Executable:", sys.executable)
    logger.info("Platform:", platform.platform())