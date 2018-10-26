# Taxi simulation

This repository contains an agent-based simulation of a taxi-passenger city system.

Current algorithm:
- Taxis are moving on a square grid.
- There is a fixed number of taxis in the system. Their velocity is 1/timestep.
- There is one taxi base in the middle of the grid.
- A fixed number of requests are generated at each timestep. The request origin and destination are generated both according predefined distributions.
- Currently, three matching algorithms are implemented (*nearest*, *poorest*, *random*). 
- If a match is made, a taxi begins to move towards the request origin, where it picks up the passenger. Then the taxi moves towards the destination, where it drops off its  passenger. After that, the taxi heads towards the base, but while it is moving back, it will be available again. The request will be marked as fulfilled.
- After a certain waiting threshold, requests are dropped.

## Configuration

The basic parameters for each simulation are stored in the `*.conf` files of the `./configs` folder. These files have a JSON structure. Here is a sample config file with comments on the parameters:

```python
{
  "n": 40, # grid width
  "m": 40, # grid height
  "price_fixed": 450, # price per trip
  "base_conf": [ # where is the taxi base
    20,
    20
  ],
  "price_per_dist": 140, # price per distance unit
  "hard_limit": 10, # max distance to be paired
  "cost_per_unit": 13, # fuel price
  "log": false, # verbose logging for debugging purposes
  "show_map_labels": false,  # taxi and request id labels on map
  "show_pending": false, # pending requests on map
  "show_plot": false, # show interactive map for debugging
  "max_request_waiting_time": 30, # max request waiting time for assignment
  "max_time": 300, # total time to run the simulation
  "batch_size": 100, # batch time after which there is a result dump
  "num_taxis": 100, # number of taxis in the system
  "request_origin_distributions": [ # origin distribution encoded as a sum of gaussians
    {
      "location": [ # 2D Gaussian mean
        15,
        15
      ],
      "strength": 5, # relative strength
      "sigma": 5 # 2D Gaussian sigma
    },
    {
      "location": [
        25,
        25
      ],
      "strength": 5,
      "sigma": 5
    }
  ],
  "avg_request_lengths": 12.5, # this is added after having generated requests according to the origin and destination distributions
  "request_rate": 10, # number of requests per time unit
  "matching": "levelling2_random_user_nearest_poorest_taxi_w_waiting_limit", # matching algorithm
  "R": 1, # request-to-taxi ratio
  "d": 200, # taxi density
  "request_destination_distributions": [ # similar to origin distributions
    {
      "location": [
        20,
        20
      ],
      "strength": 5,
      "sigma": 10
    },
    {
      "location": [
        35,
        35
      ],
      "strength": 2,
      "sigma": 2
    }
  ]
}
```

For a batch run for the same system with different parameter sets, there is always a base config file (e.g. `configs/0606_base.conf`), from which a series of config files are generated using the `generate_configs.py` file. Usage:

```
python generate_configs.py 0711_base.conf
```

## Batch run

The file `run.py` initiates one run from a config file and writes the results of the simulation to csv and json files. Usage:

```
python run.py 0525_1_priced
```

where `0525_1_priced.conf `is a file name from the `configs` directory, and the results will be saved with a similarly beginning filename into the `results` directory as a `csv` (for the aggregated metrics) and two `json` files (one for the per taxi and one for the per results metrics).

For batch running of several configurations see the manual of `./batch_run.sh`. On servers with a SLURM system, I used `./batch_slurm.sh` and `./batch_slurm_big.sh` to submit jobs to the processing queue. 

*Note: the scripts have to be marked as executable. Or you can use a `bash batch_run.sh ...` syntax.*

## Result visualization

The results of a batch run can be visualized with the following syntax:

```
python visualize.py run_pattern
```
Here, `run_pattern` is a pattern that matches the beginnings of the `run_id`s of a batch generated from the same base config file. 

This is going to generate figures into the `figs` folder.

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