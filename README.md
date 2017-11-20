# Taxi simulation

Usage from the same folder from a Jupyter Notebook, with a 10x10 city grid and 5 taxis for 5 users. Colored markers and numbers are the taxies, their path ahead is the colored line, red dots with black numbers are the users.

```
from city_models import *
%matplotlib notebook
s = Simulation(n=10,m=10,num_users=10,num_taxis=5)

s.step_time()

```

Unfortunately, only works nicely if s.step_time() is in a separate cell, and run by Ctrl+Enter multpile times by hand.
