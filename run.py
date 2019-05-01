#!/home/bokanyie/anaconda/envs/py36/bin/python

# This is a single mode run script for the simulation from terminal.
# It expects the name of a config file from the 'configs/' dir,
# without the .conf extension.
# It runs a batch simulation with settings from the config file.
# It uses the name of the config file before the .conf part as run_id.
# Outputs are written to the results/ directory.

# Example usage:
# >>> ./run.py simple


import sys
sys.path.insert(0,'/home/bokanyie/taxi') 

from city_model import *
import os

if (len(sys.argv) > 1) and (os.path.exists('configs/'+sys.argv[1]+'.conf')):
    run_id = sys.argv[1]
    config = json.load(open('configs/'+run_id+'.conf'))
    s = Simulation(**config)  # create a Simulation instance
    s.run_batch(run_id)
else:
    print('Please give an existing config file from the "./configs" folder!')

