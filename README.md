# psu-thesis-tomato-flower-classifier
A project containing the working code and data for a ResNet-based tomato flower growth stage classifier. This is for a final Honors Thesis at Penn State Harrisburg for the Schreyer's Honors College.

## Environment Vars
- Before running the script, you must create a `.env` file with your Kaggle login info.
- At `flower-classifier/process/.env`, create a file containing config for:
```.env
KAGGLE_USERNAME=
KAGGLE_KEY=
```

## Setup
1. Install `uv`
	- if you do not already have `uv` installed:
	- run `curl -LsSf https://astral.sh/uv/install.sh | sh` to install it.

2. run `bash ./setup.sh`
	- this sets up the python virtual environment and configures it for use in jupyter.
	- your terminal session should now be in that virtual environment.

3. now, run `jupyter lab --no-browser --port=8888 & disown`
	- this will start the jupyter lab program in the background
	- so the process will not terminate when the terminal session closes
	- this is very good for long-running training programs like ML tasks

4. in a browser, go to `localhost:8888/lab`
	- this page will not load
	- but in the terminal where you ran the jupyter lab command, it will print out a URL with an access token included
	- save this link for as long as you want to access that jupyter lab session
	- click that link to access the jupyter lab program.

5. Once in your jupyter notebook:
	- at the bottom of the page, click `Python`
	- change the kernel to `Python (flower-project)`
	- this allows the program to use the dependencies installations set in the venv 

6. Then, run the cells in the jupyter notebook as usual.

## Project management
- to kill the process:
	- run `pgrep jupyter` or `ps -A | grep jupyter`
	- find Jupyter's PID number
	- run `kill 1234` where 1234 is the PID number you found to stop the program.

- venv use
	- To enter an already running venv, use `source venv/bin/activate`
	- To exit from a venv, use `deactivate`

## Running the ML
- To run the ML process, first create the `.env` file and follow the `setup` instructions above.
- Then, run all of the cells in succession in `flower-classifier/process/flower_classifier.ipynb`
- There will be a log file with running outputs of ALL cells (including YOLO training process) for later reference.
	- This log file will be deleted and remade every time the `logging_config` cell is run.
