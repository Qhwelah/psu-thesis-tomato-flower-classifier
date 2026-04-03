import os, pandas, zipfile
import logging
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

path = Path(__file__).resolve().parent
while path.name != "flower-classifier" and path != path.parent:
    path = path.parent
logger.info(f"Path to project parent directory: '{path}'")
project_root = path


def get_dataset(from_kaggle:bool = False, kaggle_username:str = "", kaggle_key:str = "", google_drive_url:str = ""):
    """
    Downloads zipped dataset from kaggle or mounted google drive
    """

    download_path = project_root / 'data' / 'compressed'
    
    if from_kaggle:
        logger.info(f"Retrieving dataset from kaggle...")

        # Retrieve sign-in data 
        KAGGLE_USERNAME = kaggle_username
        KAGGLE_KEY = kaggle_key
        #kaggle.api.authenticate()
        
        # kaggle import is not top-level since importing kaggle auto-runs the kaggle.api.authenticate command
        # This is not needed if kaggle will not be used.
        import kaggle

        # Download dataset zip
        dataset_id = 'jawadulkarim117/tomato-flower-3-class'
        logger.info(f"Downloading compressed dataset via Kaggle to '{str(download_path)}'")
        
        kaggle.api.dataset_download_files(dataset_id, path=str(download_path))

    else:
        # default to google drive download
        logger.info(f"Retrieving dataset from google drive...")
        
        import gdown

        output_location = str(download_path / 'tomato-flower-3-class.zip')
        logger.info(f"Downloading compressed dataset via Google Drive to '{output_location}'")
        
        gdown.download(google_drive_url, output_location, quiet=False)


def unzip_dataset():
    """
    Extracts the contents of the zip file at the location downloaded to above.
    By default, tracks 'tomato-flower/data/compressed/tomato-flower-3-class.zip' 
    and extracts to 'tomato-flower/data/data_set'.
    """
    
    # Extract dataset zip
    path_to_zip_file = str(project_root / 'data' / 'compressed' / 'tomato-flower-3-class.zip')
    directory_to_extract_to = str(project_root / 'data' / 'data_set')
    logger.info(f"Extracting files to directory: '{directory_to_extract_to}'")


    # Create the destination directory if it doesn't exist
    if not os.path.exists(directory_to_extract_to):
        os.makedirs(directory_to_extract_to)


    # Unzip the file using a context manager (ensures the file is closed automatically)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        # Extract all contents to the specified directory
        zip_ref.extractall(directory_to_extract_to)


    logger.info(f"Files extracted successfully!")
