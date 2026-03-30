import torch
import torchvision
import ultralytics
import cv2
import yaml
import captum
import sys, platform


def print_system_info():
    """
    Finds and prints out a report of all current system info.
    """
    
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("ultralytics:", ultralytics.__version__)
    print("cv2:", cv2.__version__)
    print("captum:", captum.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())