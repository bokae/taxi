#!/home/bokanyie/anaconda3/bin/python

# Usage:
# python generate_configs.py 0711_base.conf

import sys
import json
import numpy as np
import os
from city_model import City


def avg_length(conf):

    c = City(**conf)
    tt = [c.create_one_request_coord() for i in range(c.length)]

    templ = []
    for i in range(int(len(tt)/2)):
        templ.append(np.abs(tt[2*i][0]-tt[2*i+1][0])+np.abs(tt[2*i][1]-tt[2*i+1][1]))

    return round(np.mean(templ), 1)


if __name__ == '__main__':
    os.chdir('configs')

    base = sys.argv[1]

    alg1 = "baseline_random_user_random_taxi"
    alg2 = "baseline_random_user_nearest_taxi"
    alg3 = "levelling2_random_user_nearest_poorest_taxi_w_waiting_limit"
    alg_list = [alg1, alg2, alg3]
    geom_dict = {i: json.loads(geom.strip('\n')) for i, geom
                 in enumerate(open("geom_specification_compact.json").readlines())}

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
    # three days in simulation units, supposing 8 working hours/day
    simulation_time = round(0.01*3*8*3600/tu,0)*100
    temp['max_time'] = simulation_time
    temp['batch_size'] = int(simulation_time/15)


    # =====================
    # fixed number of taxis
    # =====================

    # d = \sqrt(V/Nt)
    # fixed d=200m
    d = 200
    N = int(round(V/d**2))
    temp['num_taxis'] = N

    # different ratios
    R_list = [0.02] + list(np.linspace(0, 1, 21))[1:]
    print('R_list: ', R_list)

    for geom in geom_dict:
        temp.pop('request_origin_distribution', None)
        temp.pop('request_destination_distribution', None)
        temp.update(geom_dict[geom])
        # avg path lengths in the system
        temp['avg_request_lengths'] = avg_length(temp)

        for R in R_list:
            llambda = int(round(N*v*R / temp['avg_request_lengths'], 0))
            R_string = ('%.2f' % R).replace('.', '_')
            if llambda > 0:
                for alg in alg_list:
                    output = base.split('.')[0] + \
                        '_fixed_taxis_R_' + R_string + \
                        '_alg_' + alg + \
                        '_geom' + str(geom) + \
                        '.conf'
                    temp['request_rate'] = llambda
                    temp['matching'] = alg
                    temp['R'] = round(R, 2)
                    temp['d'] = 200

                    f = open(output, 'w')
                    f.write(json.dumps(temp, indent=4, separators=(',', ': ')))
                    f.write('\n')
                    f.close()


    # =====================
    # fixed ratio
    # =====================

    R = 0.5

    d_list = list(np.linspace(50, 400, 8))
    print('d_list: ',d_list)

    for geom in geom_dict:
        temp.pop('request_origin_distribution', None)
        temp.pop('request_destination_distribution', None)
        temp.update(geom_dict[geom])
        # avg path lengths in the system
        temp['avg_request_lengths'] = avg_length(temp)

        for d in d_list:
            N = int(round(V/d**2))
            llambda = int(round(N * v * R / temp['avg_request_lengths'], 0))
            temp['num_taxis'] = N
            temp['request_rate'] = llambda
            for alg in alg_list:
                temp['matching'] = alg
                output = base.split('.')[0] + \
                    '_fixed_ratio_N_t_' + str(N) + \
                    '_alg_' + alg + \
                    '_geom' + str(geom) + \
                    '.conf'
                temp['request_rate'] = llambda
                temp['matching'] = alg
                temp['R'] = 0.5
                temp['d'] = round(d, 0)

                f = open(output, 'w')
                f.write(json.dumps(temp, indent=4, separators=(',', ': ')))
                f.write('\n')
                f.close()

    os.chdir('..')
