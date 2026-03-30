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

- Run `system_info.ipynb`
    - Finds the versions of all the most important currently active packages
    - Also finds system details, such as which version of python is running, and if any CUDA cores are available.

- Run `make_YOLO26_config.ipynb`
    - Creates the default set of configs for YOLO26.
    - This will export into a file at `workspace/flower-classifier/config/YOLO26_config.json` by default
    - Make any changes and adjustments desired to these configs here.
    - By default, training uses `cpu` mode since this was developed on a device that did not have CUDA cores.
        - Change this setting under the `device` config.

- Run `YOLO26_classifier.ipynb`
    - This loads all of those configs into the runtime evironment, validates the system configuration, and finally runs the YOLO26 model training.
    - Manual changes to configs can also be made from within this file, right before the training cell.
    - All configs (including modified values) will be exported back to the config JSON file when the training is complete.
    - When the training is done, the system will save the training results into the `/workspace/flower-classifier/runs/` directory for later use

    - **TODO**: Add a markup or details report for each training execution that describes the configs that individual model went through.
    - **TODO**: Change this script to run detached on a background python script. 
        - This protects it from jupyter session disconnections when running from remote devices.
    - **TODO**: Create a sweep script which tests the model training with incrementally changed config sets.
        - This allows us to find more optimal configuration sets.