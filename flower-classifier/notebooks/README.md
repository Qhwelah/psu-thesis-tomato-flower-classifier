## Setup
- create `.env` file with kaggle signin info with the following attributes filled in:
```env
KAGGLE_USERNAME=
KAGGLE_KEY=
```

- run the `get_dataset_local.ipynb` jupyter script to pull the dataset from Kaggle

- if you wish, run the `view_images.ipynb` file to verify successful download.

- Then, run the `ingest.ipynb` script to set up the project directory for YOLO26 to run on.
    - this also creates a `data.yaml` file which labels the directory for YOLO26 to use.