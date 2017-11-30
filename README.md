# Taxi simulation

Usage from the same folder from a Jupyter Notebook.


Importing the classes, initializing interactive plotting, importing the timestep button.

```
from city_model import *
%matplotlib notebook
from ipywidgets import Button
```

Configuring the simulation, then presenting it visually:
```
config=dict(
    n=40, # width of grid
    m=20, # height of grid
    num_taxis=5, # total number of taxis
    request_rate=0.1, # request probability per time unit
    base_sigma=5, # Gauss-std in grid units
    hard_limit=10, # do not serve requests that are that far
    show_map_labels=False, # show taxi and request numbers
    log=False, # be verbose about events in print log
    show_pending=True # show pendng requests on map
)
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
* We begin to match our requests to the taxis according to the matching algorithm.
* If a match is made, a taxi begins to move towards the request origin, where it picks up the passenger. Then the taxi moves towards the destination, where it drops off its  passenger. After that, the taxi heads towards the base, but while it is moving back, it will be available again.
