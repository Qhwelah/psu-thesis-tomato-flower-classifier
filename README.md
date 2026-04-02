# psu-thesis-tomato-flower-classifier
A project containing the working code and data for a ResNet-based tomato flower growth stage classifier. This is for a final Honors Thesis at Penn State Harrisburg for the Schreyer's Honors College.

## Setup
1. run `bash ./setup.sh`
	- this sets up the python virtual environment and configures it for use in jupyter.
	- your terminal session should now be in that virtual environment.

2. now, run `jupyter lab --no-browser --port=8888 & disown`
	- this will start the jupyter lab program in the background
	- so the process will not terminate when the terminal session closes
	- this is very good for long-running training programs like ML tasks

3. in a browser, go to `localhost:8888/lab`
	- this page will not load
	- but in the terminal where you ran the jupyter lab command, it will print out a URL with an access token included
	- save this link for as long as you want to access that jupyter lab session
	- click that link to access the jupyter lab program.

4. Once in your jupyter notebook:
	- at the bottom of the page, click `Python`
	- change the kernel to `Python (flower-project)`
	- this allows the program to use the dependencies installations set in the venv 

5. Then, run the cells in the jupyter notebook as usual.

## Project management
- to kill the process:
	- run `pgrep jupyter` or `ps -A | grep jupyter`
	- find Jupyter's PID number
	- run `kill 1234` where 1234 is the PID number you found to stop the program.

- venv use
	- To enter an already running venv, use `source venv/bin/activate`
	- To exit from a venv, use `deactivate`
