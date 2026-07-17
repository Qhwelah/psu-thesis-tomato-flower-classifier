## Setup
- create `.env` file with kaggle sign-in info with the following attributes filled in:
```env
KAGGLE_USERNAME=
KAGGLE_KEY=
```

## Modules
- `get_dataset.py`
    - jupyter script to pull the dataset from Kaggle

- `view_file.py`
    - functions used to verify successful download and file/picture viewing.

- `ingest.py`
    - script to set up the project directory for YOLO26 to run on.
    - this also creates a `data.yaml` file which labels the directory for YOLO26 to use.

- `system_info.py`
    - Finds the versions of all the most important currently active packages
    - Also finds system details, such as which version of python is running, and if any CUDA cores are available.

- `config_IO.py`
    - Functions for reading and writing config setups to a JSON file for ease of use.
    - Also has a function to create the default set of configs for YOLO26.
        - This will export into a file at `workspace/flower-classifier/config/YOLO26_config.json` by default
        - Make any changes and adjustments desired to these configs here.
        - By default, training uses `cpu` mode since this was developed on a device that did not have CUDA cores.
            - Change this setting under the `device` config.

- `flower_classifier.ipynb`
    - This loads all of those configs into the runtime evironment, validates the system configuration, and finally runs the YOLO26 model training.
    - Manual changes to configs can also be made from within this file, right before the training cell.
    - All configs (including modified values) will be exported back to the config JSON file when the training is complete.
    - When the training is done, the system will save the training results into the `/workspace/flower-classifier/runs/` directory for later use

    - **TODO**: Add a markup or details report for each training execution that describes the configs that individual model went through.
    - **TODO**: Create a sweep script which tests the model training with incrementally changed config sets.
        - This allows us to find more optimal configuration sets.