# psu-thesis-tomato-flower-classifier
A project containing the working code and data for a ResNet-based tomato flower growth stage classifier. This is for a final Honors Thesis at Penn State Harrisburg for the Schreyer's Honors College.

## Setup
- run `bash ./setup.sh`
	- this sets up the python virtual environment and configures it for use in jupyter.
	- your terminal session should now be in that virtual environment.

- now, run `jupyter lab --no-browser --port=8888 & disown`
	- this will start the jupyter lab program in the background
	- so the process will not terminate when the terminal session closes
	- this is very good for long-running training programs like ML tasks

- to kill the process:
	- run `pgrep jupyter` or `ps -A | grep jupyter`
	- find Jupyter's PID number
	- run `kill 1234` where 1234 is the PID number you found to stop the program.

- venv use
	- To enter an already running venv, use `source venv/bin/activate`
	- To exit from a venv, use `deactivate`
