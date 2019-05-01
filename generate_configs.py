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


class ConfigGenerator:
    """
    :param base: file name for common configuration parameters
    :param days: number of real days the simulation should run
    """

    def __init__(self, base, days=5):

        self.base = base

        # different matching algorithms
        alg1 = "random_unlimited"
        alg2 = "random_limited"
        alg3 = "nearest"
        alg4 = "poorest"
        self.alg_list = [alg1, alg2, alg3, alg4]

        # different geometries
        geom_dict_all = {i: json.loads(geom.strip('\n')) for i, geom
                         in enumerate(open("configs/geom_specification_compact.json").readlines())}

        self.geom_dict = {i: geom_dict_all[i] for i in geom_dict_all.keys()}

        # common parameters
        self.common = json.load(open('configs/' + base))

        # =====================
        # global parameters
        # =====================

        # 1 distance unit in meters
        self.scale = 100

        # system volume
        self.V = self.common['n']*self.common['m']*self.scale**2

        # velocity of taxis in distance unt per time unit
        # should correspond to 36 km/h!!!
        self.v = 1
    
        # time unit in seconds
        self.tu = self.scale/10*self.v
    
        # three days in simulation units, supposing 8 working hours/day
        simulation_time = round(0.01*days*8*3600/self.tu, 0)*100
        self.common['max_time'] = simulation_time
        self.common['batch_size'] = int(simulation_time/(days*16)) # 16 sample points in each shift
    
        # reset taxi positions after an 8-hour shift
        self.reset_time = round(0.01*8*3600/self.tu, 0)*100

        self.behav_types = [("go_back", "base", "false"),
            ("stay", "base", "false"),
            ("stay", "home", "false"),
            ("stay", "home", "true")]

    @staticmethod
    def avg_length(conf):
        """
        Given a configuration dictionary, calculates average request length in a geometry.
        """

        c = City(**conf)
        tt = [c.create_one_request_coord() for i in range(c.length)]

        templ = []
        for i in range(int(len(tt) / 2)):
            templ.append(np.abs(tt[2 * i][0] - tt[2 * i + 1][0]) + np.abs(tt[2 * i][1] - tt[2 * i + 1][1]))

        return round(np.mean(templ), 1)

    def generate_config(self, d, R, alg, geom, behav_type):
        """
        Parameters
        ----------
        d : 
        R : 
        alg : 
        geom : 
        behav_type : 

        Returns
        -------

        """
        
        conf = self.common
        
        conf.pop('request_origin_distributions', None)
        conf.pop('request_destination_distributions', None)

        conf.update(self.geom_dict[geom])
        conf['geom'] = geom

        # parameters
        conf['R'] = round(R, 2)
        conf['d'] = round(d, 0)

        # d = \sqrt(V/Nt)
        N = int(round(self.V / d ** 2))
        conf['num_taxis'] = N

        conf['avg_request_lengths'] = self.avg_length(conf)

        llambda = int(round(N * self.v * R / conf['avg_request_lengths'], 0))
        if llambda == 0:
            return

        conf['request_rate'] = llambda



        if type(alg)==int:
            conf['matching'] = self.alg_list[alg]
        else:
            conf["matching"] = alg

        behaviour, ic, reset = self.behav_types[behav_type]
        conf.update({"behaviour": behaviour, "initial_conditions": ic})
        conf['reset'] = reset
        if reset == 'true':
            conf.update({"reset_time": g.reset_time})
        else:
            conf.pop("reset_time", None)

        return conf

    def dump_config(self, conf):
        # pop non JSON-serializable element
        for k in ["request_origin_distributions", "request_destination_distributions"]:
            if k in conf:
                for elem in conf[k]:
                    elem.pop("cdf_inv", None)

        # request rate
        R_string = ('%.2f' % conf['R']).replace('.', '_')
        d_string = '%d' % conf['d']

        # filename
        fname = self.base.split('.')[0] + \
             '_d_' + d_string + \
             '_R_' + R_string + \
             '_alg_' + conf['matching'] + \
             '_geom_' + str(conf['geom']) + \
             '_behav_' + conf['behaviour'] + \
             '_ic_' + conf['initial_conditions'] + \
             '_reset_' + conf['reset'] + \
             '.conf'

        content = json.dumps(conf, indent=4, separators=(',', ': ')) + '\n'

        return fname, content


if __name__ == '__main__':

    mode = sys.argv[1]

    if mode == "sweep":

        "Generating configs for all possible config combinations for exploration purposes."

        g = ConfigGenerator(sys.argv[2])

        # ====================================================
        # generate configs corresponding to parameter matrix
        # ====================================================

        # different Gaussian geoms
        for geom in range(7):
            for behav_type in range(len(g.behav_types)):
                # sweeping through a range of R and d systematically
                d_list = list(np.linspace(50, 400, 11))
                for d in d_list:
                    # different ratios
                    R_list = list(np.linspace(0.05, 1, 20))
                    for R in R_list:
                            # inserting different algorithms
                            for alg in g.alg_list:

                                conf = g.generate_config(d, R, alg, geom, behav_type)
                                fname,content = g.dump_config(conf)

                                # dump
                                f = open('configs/' + fname, 'w')
                                f.write(content)
                                f.close()

    elif mode == "long_run":
        gen = ConfigGenerator('2019_02_14_base.conf',days=100)
        conf = gen.generate_config(225, 0.5, 'nearest', 0, 1)
        fname, content = gen.dump_config(conf)
        fname = fname.split('.')[0] + '_long_run.conf'
        # dump
        f = open('configs/' + fname, 'w')
        f.write(content)
        f.close()

    elif mode == "new_geoms":
        gen = ConfigGenerator('2019_02_14_base.conf')
        geoms = [0,7,8,9]
        for g in geoms:
            print(g)
            conf = gen.generate_config(225, 0.5, 'nearest', g, 1)
            fname, content = gen.dump_config(conf)
            fname = fname.split('.')[0] + '_new_geoms.conf'
            # dump
            f = open('configs/' + fname, 'w')
            f.write(content)
            f.close()

    elif mode == "multiple_runs":
        print()
