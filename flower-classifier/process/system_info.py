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
    
    logger.info(f" \
        \n torch {torch.__version__} \
        \n torchvision {torchvision.__version__} \
        \n ultralytics {ultralytics.__version__} \
        \n cv2 {cv2.__version__} \
        \n captum {captum.__version__} \
        \n CUDA available: {torch.cuda.is_available()} \
        \n Python {sys.version} \
        \n Executable {sys.executable} \
        \n Platform {platform.platform()} \
        \n")