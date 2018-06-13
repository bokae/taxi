# Taxi simulation

This repository contains an agent-based simulation of a taxi-passenger city system.

Current algorithm:
* Taxis are moving on a square grid.
* There is a fixed number of taxis in the system. Their velocity is 1/timestep.
* There is one taxi base in the middle of the grid.
* A fixed number of requests are generated at each timestep. The request origin and destination are generated both according to a 2D Gauss spatial distribution around the taxi base. The distribution has rotational symmetry (but it would be easy to change).
* Currently, three matching algorithms are implemented. 
* If a match is made, a taxi begins to move towards the request origin, where it picks up the passenger. Then the taxi moves towards the destination, where it drops off its  passenger. After that, the taxi heads towards the base, but while it is moving back, it will be available again. The request will be marked as fulfilled.
* After a certain waiting threshold, requests are dropped.

## Configuration

The basic parameters for each simulation are stored in the `*.conf` files of the `./configs` folder. These files have a JSON structure. Here is a sample config file with comments on the parameters:

```
{
    "n":10, # grid width
    "m":10, # grid height
    "num_taxis":5, # number of taxis
    "request_rate":1, # number of requests per timestep
    "log":true, # verbose logging of events to stdout
    "max_time":1000, # simulation overall runtime
    "base_sigma":5, # std of gauss around the tax base
    "hard_limit":10, # search for nearest raxis in this circle
    "show_map_labels":true, # show taxi and request ids on interactive plot
    "show_pending":true, # show pending requests on interactive plot
    "show_plot":true, # show interactive plot
    "matching":"baseline_random_user_nearest_taxi" # name of matching algorithm
}

```

For a batch run for the same system with different parameter sets, there is always a base config file (e.g. `configs/0606_base.conf`), which can then be easily extended by the `configs/add_param.sh` script. Usage:

```
cd configs
cat 0606_base.conf | ./add_param.sh -n num_taxis 2500
```
See the manual of `add_param.sh` for further details.

To generate a set of config files for parameter ranges in one script, see the manual of `generate_configs.sh`.

# Batch run

The file `run.py` initiates runs from config files and writes the results of the simulation to a csv and a json file. Usage is 

```
./run.py 0525_1_priced
```

where `0525_1_priced.conf `is a file name from the `configs` directory, and the results will be saved with a similarly beginning filename into the `results` directory as a `csv` (for the aggregated metrics) and two `json` files (one for the per taxi and one for the per results metrics).

*Note: maybe you have to modify the path in the first line of the script to make it point to your favourite Python3 interpreter. The script has to be marked as executable for such usage. Or you can use a `python run.py ...` syntax.*

For batch running of several configurations see the manual of `./batch_run.sh`. On servers with a SLURM system, I used `./batch_slurm.sh` and `./batch_slurm_big.sh` to submit jobs to the processing queue. 

*Note: the scripts have to be marked as executable. Or you can use a `bash batch_run.sh ...` syntax.*

## Result visualization

The results of a batch run can be visualized with the following syntax:

```
python visualize.py run_pattern
```
Here, `run_pattern` is a pattern that matches the beginnings of the `run_id`s of a batch generated from the same base config file. Currently, two results are accessible with the following commands:

```
python visualize.py 0606_base_f
python visualize.py 0609_base_big
```

## Debugging with interactive visualization

For debugging and fun purposes, display the map of the simulation with the taxis and requests color-coded and moving. Must be run from a Juypter Notebook.

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

We can print the state of taxis or requests with the usual print command using their IDs. For example:

```
print(s.requests[3])
print(s.taxis[1])
```