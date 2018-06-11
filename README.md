# Taxi simulation

For debugging and fun purposes, display the map of the simulation with the taxis and requests color-coded and moving. Must be run from a Juypter Notebook. Due to some bugs in interactivity, sometimes the code cells have to be run several times before the simulation displays.


Importing the classes, initializing interactive plotting, importing the timestep button.

```
%matplotlib notebook
from city_model import *
from ipywidgets import Button
import json
```

Configuring the simulation in a JSON file, then displaying the map.
```
config=json.load(open('configs/simple.conf'))
s = Simulation(**config) # create a Simulation instance
b = Button(description = "Step_time") # create clickable time tick button
b.on_click(s.step_time) # assign time ticker function to button callback
b
```

For debugging purposes, we can switch plotting, logging, certain annotations on the plot on/off.

We can print the state of taxis or requests with the usual print command using their IDs. For example:
```
print(s.requests[3])
print(s.taxis[1])
```

Current algorithm:
* There is a fixed number of taxis in the system. Their velocity is 1/timestep.
* There is a taxi base in the middle of the grid.
* Requests are generated with a fixed probability at each timestep. The request origin and destination are generated both according to a 2D Gauss spatial distribution around the taxi base. The distribution has rotational symmetry (but it would be easy to change). The standard deviation of this distribution is fiven by `base_sigma`.
* Matching algorithms can be controlled via the `matching` keyword of the configuration. For more details, see the docstring of the `Simulation.matching_algorithm()` method.  
* If a match is made, a taxi begins to move towards the request origin, where it picks up the passenger. Then the taxi moves towards the destination, where it drops off its  passenger. After that, the taxi heads towards the base, but while it is moving back, it will be available again.

# Batch run

The file `run.py` initiates runs from config files and writes the results of the simulation to a csv and a json file. Usage is 

```
./run.py 0525_1_priced
```

where `0525_1_priced.conf `is a file name from the `configs` directory, and the results will be saved with a similarly beginning filename into the `results` directory as a `csv` (for the aggregated metrics) and two `json` files (one for the per taxi and one for the per results metrics).

*Note: maybe you have to modify the path in the first line of the script to make it point to your favourite Python3 interpreter. The script has to be marked as executable for such usage. Or you can use a `python run.py ...` syntax.*

# Shell scripts

There are three additional shell scripts that help config parameter editing (`configs/add_param.sh`), batch config file generation (`configs/generate_params.sh`) and batch running of several configurations (`./batch_run.sh`). See the scripts for further documentation.

*Note: the scripts have to be marked as executable. Or you can use a `bash batch_run.sh ...` syntax.*