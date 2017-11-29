# Taxi simulation

Usage from the same folder from a Jupyter Notebook, with a 10x10 city grid and 5 taxis for 5 users. Colored markers and numbers are the taxis, their path ahead is the colored line, red dots with black numbers are the users.

```
from city_model import *
%matplotlib notebook
from time import sleep

config=dict(n=10,m=10,num_taxis=10,request_rate=1,base_sigma=3,hard_limit=10)
s = Simulation(**config)

for i in range(100):
    s.step_time()
```
```

