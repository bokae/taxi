#!/home/bokanyie/anaconda3/bin/python

# Usage:
# python generate_configs.py 0711_base.conf
#
# This script generates config files for runs with different parameters. It creates a sweep of parameters R and d based
# on the geometry settings. The geometries are defined in the geom_specification_compact.json file, where each line
# defines the distribution of request origins and destinations as the superposition of 2D Gaussian distributions.

import sys
import json
import numpy as np
import os
from city_model import City


def avg_length(conf):
    """
    Given a configuration dictionary, calculates average request length in a geometry.
    """

    c = City(**conf)
    tt = [c.create_one_request_coord() for i in range(c.length)]

    templ = []
    for i in range(int(len(tt)/2)):
        templ.append(np.abs(tt[2*i][0]-tt[2*i+1][0])+np.abs(tt[2*i][1]-tt[2*i+1][1]))

    return round(np.mean(templ), 1)


if __name__ == '__main__':
    os.chdir('configs')

    base = sys.argv[1]

    # different matching algorithms
    #alg1 = "baseline_random_user_random_taxi"
    #alg2 = "baseline_random_user_nearest_taxi"
    #alg3 = "levelling2_random_user_nearest_poorest_taxi_w_waiting_limit"
    #alg_list = [alg1, alg2, alg3]

    alg_list = ["random_unlimited"]

    # different geometries
    geom_dict_all = {i: json.loads(geom.strip('\n')) for i, geom
                 in enumerate(open("geom_specification_compact.json").readlines())}

    geom_dict = {i: geom_dict_all[i] for i in [4,5]}

    # common parameters
    temp = json.load(open(base))

    # =====================
    # global parameters
    # =====================

    # 1 distance unit in meters
    scale = 100

    # system volume
    V = temp['n']*temp['m']*scale**2

    # velocity of taxis in distance unt per time unit
    # should correspond to 36 km/h!!!
    v = 1

    # time unit in seconds
    tu = scale/10*v

    # five days in simulation units, supposing 8 working hours/day
    simulation_time = round(0.01*5*8*3600/tu, 0)*100
    temp['max_time'] = simulation_time
    temp['batch_size'] = int(simulation_time/80) # 16 sample points in each shift

    # reset taxi positions after an 8-hour shift
    reset_time = round(0.01*8*3600/tu, 0)*100

    # ====================================================
    # generate configs corresponding to parameter matrix
    # ====================================================

    for geom in sorted(geom_dict.keys()):
        # inserting different geometries into the config dict
        temp.pop('request_origin_distributions', None)
        temp.pop('request_destination_distributions', None)
        temp.pop('reset_time', None)
        temp.update(geom_dict[geom])

        # avg path lengths in the system
        temp['avg_request_lengths'] = avg_length(temp)

#            ("go_back", "base", "false"),
#            ("stay", "base", "false"),
#            ("stay", "home", "false"),

        for behaviour, ic, reset in [
            ("stay", "home", "true")
        ]:
            temp.update({"behaviour": behaviour, "initial_conditions": ic})

            if reset == 'true':
                temp.update({"reset_time": reset_time})

            # sweeping through a range of R and d systematically

            # d = \sqrt(V/Nt)
            d_list = list(np.linspace(50, 400, 11))
            for d in d_list:
                N = int(round(V/d**2))
                temp['num_taxis'] = N

                # different ratios
                R_list = list(np.linspace(0.05, 1, 20))
                for R in R_list:
                    # request rate
                    llambda = int(round(N*v*R / temp['avg_request_lengths'], 0))
                    R_string = ('%.2f' % R).replace('.', '_')
                    d_string = '%d' % d

                    # parameters
                    temp['request_rate'] = llambda
                    temp['R'] = round(R, 2)
                    temp['d'] = round(d, 0)

                    if llambda > 0:
                        # inserting different algorithms
                        for alg in alg_list:
                            temp['matching'] = alg

                            # filename
                            output = base.split('.')[0] + \
                                '_d_' + d_string + \
                                '_R_' + R_string + \
                                '_alg_' + alg + \
                                '_geom_' + str(geom) + \
                                '_behav_' + behaviour + \
                                '_ic_' + ic + \
                                '_reset_' + reset + \
                                '.conf'

                            # dump
                            f = open(output, 'w')
                            f.write(json.dumps(temp, indent=4, separators=(',', ': ')))
                            f.write('\n')
                            f.close()

    os.chdir('..')
