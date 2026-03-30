import os, kaggle, pandas, zipfile
from dotenv import load_dotenv


# Retrieve sign-in data 
load_dotenv()
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
kaggle.api.authenticate()


# Download dataset zip
dataset_id = 'jawadulkarim117/tomato-flower-3-class'
download_path = '/workspace/data/compressed'
kaggle.api.dataset_download_files(dataset_id, path=download_path)


# Extract dataset zip
path_to_zip_file = download_path + '/tomato-flower-3-class.zip'
directory_to_extract_to = '/workspace/data/data_set'


# Create the destination directory if it doesn't exist
if not os.path.exists(directory_to_extract_to):
    os.makedirs(directory_to_extract_to)


# Unzip the file using a context manager (ensures the file is closed automatically)
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    # Extract all contents to the specified directory
    zip_ref.extractall(directory_to_extract_to)


print(f"Files extracted to: {directory_to_extract_to}")
