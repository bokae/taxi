#!/home/bokanye/anaconda3/bin/python

import sys
import json
import numpy as np
import os
from city_model import City

os.chdir('configs')


def avg_length(conf):

    c = City(**conf)
    tt = c.generate_coords()

    templ = []
    for i in range(int(len(tt)/2)):
        templ.append(np.abs(tt[2*i][0]-tt[2*i+1][0])+np.abs(tt[2*i][1]-tt[2*i+1][1]))

    return round(np.mean(templ), 1)


base = sys.argv[1]

alg1 = "baseline_random_user_random_taxi"
alg2 = "baseline_random_user_nearest_taxi"
alg3 = "levelling2_random_user_nearest_poorest_taxi_w_waiting_limit"
alg_list = [alg1, alg2, alg3]

temp = json.load(open(base))

# =====================
# global parameters
# =====================

# 1 distance unit in meters
scale = 100
# system volume
V = temp['n']*temp['m']*scale**2
# velocity of taxis in distance unt per time unit
v = 1
# avg path lengths in the system
l = avg_length(temp)

# =====================
# fixed number of taxis
# =====================

# d = \sqrt(V/Nt)
# fixed d=200m
d = 200
N = int(round(V/d**2))
temp['num_taxis'] = N

# different ratios
R_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for R in R_list:
    llambda = int(round(N*v*R/l, 0))
    R_string = ('%.1f' % R).replace('.', '_')
    if llambda > 0:
        for alg in alg_list:
            output = base.split('.')[0]+'_fixed_taxis_r_'+R_string+'_alg_'+alg+'.conf'
            temp['request_rate'] = llambda
            temp['matching'] = alg
            f = open(output, 'w')
            f.write(json.dumps(temp,indent=4,separators=(',', ': ')))
            f.write('\n')
            f.close()


# =====================
# fixed ratio
# =====================

R = 0.5

d_list = [50, 100, 200, 300, 400, 500, 750, 1000]

for d in d_list:
    N = int(round(V/d**2))
    llambda = int(round(N * v * R / l, 0))
    temp['num_taxis'] = N
    temp['request_rate'] = llambda
    for alg in alg_list:
        temp['matching'] = alg
        output = base.split('.')[0] + '_fixed_ratio_t_' + str(N) + '_alg_' + alg + '.conf'
        temp['request_rate'] = llambda
        temp['matching'] = alg
        f = open(output, 'w')
        f.write(json.dumps(temp, indent=4, separators=(',', ': ')))
        f.write('\n')
        f.close()


os.chdir('..')
