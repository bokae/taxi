#!/home/bokanyie/anaconda3/bin/python
# This is a batch run script for the simulation from terminal.
# It expects the name of a config file from the 'configs/' dir.
# It runs a batch simulation with setting from the config file.
# It uses the name of the config file before the .con part as run_id.


from city_model import *
import sys
import os

if (len(sys.argv)>1) and (os.path.exists('configs/'+sys.argv[1])):
    run_id=sys.argv[1].split(".")[0]
    config=json.load(open('configs/'+run_id+'.conf'))
    s = Simulation(**config) # create a Simulation instance
    s.run_batch(run_id)
else:
    print('Please give an existing config file from the "./configs" folder!')

