from IPython.display import Image, display
from pathlib import Path


def view_image(path: str | Path):
    try:
        display(Image(path))
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def view_text(path: str | Path):
    try:
        with open(path, 'r') as file:
            file_contents = file.read()
            print(file_contents)
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")