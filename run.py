#!/home/bokanyie/anaconda/envs/py36/bin/python

# This is a single mode run script for the simulation from terminal.
# It expects the name of a config file from the 'configs/' dir,
# without the .conf extension.
# It runs a batch simulation with settings from the config file.
# It uses the name of the config file before the .conf part as run_id.
# Outputs are written to the results/ directory.

# Example usage:
# >>> ./run.py simple


from city_model import *
import os
import sys

if (len(sys.argv) > 1) and (os.path.exists(sys.argv[1])):
    p = "/".join(sys.argv[1].split("/")[1:][:-1])
    run_id = sys.argv[1].split("/")[-1].split(".")[0]
    config = json.load(open(sys.argv[1]))
    s = Simulation(**config)  # create a Simulation instance
    s.run_batch(run_id, data_path="results/"+p)
else:
    print('Please give an existing config file from the "./configs" folder!')

#TODO modify all run scripts, because config reference has changed!

