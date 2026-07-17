from IPython.display import Image, display
from pathlib import Path
import logging

logger = logging.getLogger(f"flower_classifier.{__name__}")

def view_image(path: str | Path):
    try:
        display(Image(path))
    except FileNotFoundError:
        logger.info(f"Error: The file '{path}' was not found.")
    except Exception as e:
        logger.info(f"An error occurred: {e}")


def view_text(path: str | Path):
    try:
        with open(path, 'r') as file:
            file_contents = file.read()
            logger.info(file_contents)
    except FileNotFoundError:
        logger.info(f"Error: The file '{path}' was not found.")
    except Exception as e:
        logger.info(f"An error occurred: {e}")