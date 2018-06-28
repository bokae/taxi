#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:00:15 2017

@author: bokanyie
"""

# Benchmark config file for runtime: 0525_1_priced.conf
# Benchmark run: python run.py 0525_1_priced
# Benchmark batch time average original: 14.438
# Benchmark batch time average with RandomDict: 13.197, not much better...
# Benchmark batch time after using more list comprehensions and filters: 12.045, not much better...
# Benchmark batch time after generating coordinates in advance:  5.514, really great!
# Benchmark batch time after fixing last variable in move_taxi: 5.115, not much gain.
# Storing the neighbors in advance slightly improved performance.

import numpy as np
import pandas as pd
import json
from random import shuffle, choice, gauss
import matplotlib.pyplot as plt
from time import time
from scipy.stats import entropy

# special data types
from collections import deque
from queue import Queue
from randomdict import RandomDict


class City:
    """
    Represents a grid on which taxis are moving.
    """

    def __init__(self, **config):
        """      
        Parameters
        ----------
        
        n : int
            width of grid
        
        m : int
            height of grid

        base_coords : [int,int]
            grid coordinates of the taxi base
            
        base_sigma : float
            standard deviation of the 2D Gauss distribution
            around the taxi base

        Attributes
        ----------
        A : np.array of lists
            placeholder to store list of available taxi_ids at their actual
            positions

        n : int
            width of grid
        
        m : int
            height of grid

        base_coords: [int,int]
            coordinates of the taxi base
            
        base_sigma: float
            standard deviation of the 2D Gauss distribution
            around the taxi base
                    
        
        
        """
        if ("n" in config) and ("m" in config):
            self.n = config["n"]  # number of pixels in x direction
            self.m = config["m"]  # number of pixels in y direction

            # array that stores taxi_id of available taxis at the
            # specific position on the grid
            # we initialize this array with empy lists
            self.A = np.empty((self.n, self.m), dtype=set)
            for i in range(self.n):
                for j in range(self.m):
                    self.A[i, j] = set()

            self.N = np.empty((self.n, self.m), dtype=set)
            for i in range(self.n):
                for j in range(self.m):
                    self.N[i, j] = self.neighbors((i, j))

            self.base_coords = [int(np.floor(self.n / 2) - 1), int(np.floor(self.m / 2) - 1)]
            #            print(self.base_coords)
            self.base_sigma = config["base_sigma"]

    def measure_distance(self, source, destination):
        """
        Measure distance on the grid between two points.
        
        Returns
        -------
        Source coordinates are marked by *s*,
        destination coordinates are marked by *d*.
        
        The distance is the following integer:
        $$|x_s-x_d|+|y_s-y_d|$$
        """

        return np.dot(np.abs(np.array(destination) - np.array(source)), [1, 1])

    def create_path(self, source, destination):
        """
        Choose a random shortest path between source and destination.
        
        Parameters
        ----------
        
        source : [int,int]
            grid coordinates of the source
            
        destination : [int,int]
            grid coordinates of the destination
        
        
        Returns
        -------
        
        path : list of coordinate tuples
            coordinate list of a random path between source and destinaton
            
        """

        # distance along the x and the y axis
        d = dict(zip(['x','y'],np.array(destination) - np.array(source)))

        # create a sequence of "x"-es and "y"-s
        # we are going to shuffle this sequence
        # to get a random order of "x" and "y" direction steps
        sequence = ['x'] * int(np.abs(d['x'])) + ['y'] * int(np.abs(d['y']))
        shuffle(sequence)

        # source is included in the path
        path = [source]
        for item in sequence:
                # we add one step in the right direction based on the last position
                path.append([
                    np.sign(d[item])*int(item=="x") + path[-1][0],
                    np.sign(d[item])*int(item=="y") + path[-1][1]
                ])

        return path

    def neighbors(self, coordinates):
        """
        Calculate the neighbors of a coordinate.
        On the edges of the simulation grid, there are no neighbors.
        (E.g. there are only 2 neighbors in the corners.)

        Parameters
        ----------
        
        coordinates : [int,int]
            input grid coordinate
            
        Returns
        -------
        
        ns : list of coordinate tuples
            list containing the coordinates of the neighbors        
        """

        ns = [(coordinates[0] + dx, coordinates[1] + dy) for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        ns = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]),ns)

        return set(ns)


    def create_request_coords(self, mean=None):
        """
        Creates random request coords based on the base_coords and the
        base_sigma according to a 2D Gauss.
        
        Returns
        -------
        
        x,y
            coordinate tuple
        """

        done = False
        if mean is None:
            mean = self.base_coords

        cov = np.array([[self.base_sigma ** 2, 0], [0, self.base_sigma ** 2]])

        while not done:

            x, y = np.random.multivariate_normal(mean, cov)

            x = int(np.floor(x))
            y = int(np.floor(y))

            if (x >= 0) and (x < self.n) and (y >= 0) and (y < self.m):
                done = True

        return x, y

    def generate_coords(self,length=None):

        if length is None:
            length=2e5

        temp = map(
                lambda t:(int(round(t[0],0)),int(round(t[1],0))),
                   [(self.base_coords[0]+gauss(0,self.base_sigma),self.base_coords[1]+gauss(0,self.base_sigma)) for i in range(int(length))])
        temp = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]),temp)

        return deque(temp)


class Taxi:
    """
    Represents a taxi in the simulation.
    
    Attributes
    ----------
    
    x : int
        horizontal grid coordinate
    
    y : int
        vertical grid coordinate
    
    taxi_id : int
        unique identifier of taxi
        
    available : bool
        flag that stores whether taxi is free
        
    to_request : bool
        flag that stores when taxi is moving towards a request
        but there is still no user sitting in it
    
    with_passenger : bool
        flag that stores when taxi is carrying a passenger
        
    actual_request_executing : int
        id of request that is being executed by the taxi
        
    requests_completed : list of ints
        list of requests completed by taxi
        
    time_waiting : int
        time spent with empty waiting
    
    time_serving : int
        time spent with carrying a passenger

    time_to_request : int
        time spent from call assignment to pickup, without a passenger

    time_cruising : int
        time spent with travelling empty with no assigned requests

    next_destination : Queue
        Queue that stores the path forward of the taxi
    
    """

    def __init__(self, coords=None, taxi_id=None):
        if coords is None:
            print("You have to put your taxi somewhere in the city!")
        elif taxi_id is None:
            print("Not a licenced taxi.")
        else:

            self.x = coords[0]
            self.y = coords[1]

            self.taxi_id = taxi_id

            self.available = True
            self.to_request = False
            self.with_passenger = False

            self.actual_request_executing = None
            self.requests_completed = set()

            # types of time metrics to be stored
            self.time_waiting = 0
            self.time_serving = 0
            self.time_cruising = 0
            self.time_to_request = 0

            # storing steps to take
            self.next_destination = Queue()  # path to travel

    def __str__(self):
        s = [
            "Taxi ",
            str(self.taxi_id),
            ".\n\tPosition ",
            str(self.x) + "," + str(self.y) + "\n"
        ]
        if self.available:
            s += ["\tAvailable.\n"]
        elif self.to_request:
            s += ["\tTravelling towards request " + str(self.actual_request_executing) + ".\n"]
        elif self.with_passenger:
            s += ["\tCarrying the passenger of request " + str(self.actual_request_executing) + ".\n"]

        return "".join(s)


class Request:
    """
    Represents a request that is being made.
    
    Attributes
    ----------
    
    ox,oy : int
        grid coordinates of request origin
        
    dx,dy : int
        grid coordinates of request destination
        
    request_id : int
        unique id of request
    
    request_timestamp : int
        timestamp of request
        
    pickup_timestamp : int
        timestamp of taxi arrival
        
    dropoff_timestamp : int
        timestamp of request completed
    
    taxi_id : int
        id of taxi that serves the request
    
    waiting_time : int
        how much time the user had to wait until picked up
    """

    def __init__(self, ocoords=None, dcoords=None, request_id=None, timestamp=None):
        """
        
        """
        if (ocoords is None) or (dcoords is None):
            print("A request has to have a well-defined origin and destination.")
        elif request_id is None:
            print("Please indentify each request uniquely.")
        elif timestamp is None:
            print("Please give a timestamp for the request!")
        else:
            # pickup coordinates
            self.ox = ocoords[0]
            self.oy = ocoords[1]

            # desired dropoff coordinates
            self.dx = dcoords[0]
            self.dy = dcoords[1]

            # id
            self.request_id = request_id

            # travel info, different time metrics
            self.taxi_id = None

            self.timestamps = {
                'request' : timestamp,
                'pickup' : None,
                'dropoff' : None
            }

            self.mode='pending'

            self.time = {
                'pending' : 0,
                'waiting' : 0,
                'serving' : 0
            }

    def timer(self):
        if self.mode!='done' and self.mode!='dropped':
            self.time[self.mode]+=1

    def __str__(self):
        s = [
            "Request ",
            str(self.request_id),
            ".\n\tOrigin ",
            str(self.ox) + "," + str(self.oy) + "\n",
            "\tDestination ",
            str(self.dx) + "," + str(self.dy) + "\n",
            "\tRequest timestamp ",
            str(self.timestamps['request']) + "\n"
        ]
        if self.taxi_id is not None:
            s += ["\tTaxi assigned ", str(self.taxi_id), ".\n"]
            s += ["\tWaiting since ", str(self.time['waiting']), ".\n"]
            if self.time['serving'] != 0:
                s += ["\tPickup timestamp ", str(self.timestamps['pickup']), ".\n"]
                s += ["\tServing since ", str(self.time['serving']), ".\n"]
                if self.timestamps['dropoff'] is not None:
                    s += ["\tDropoff timestamp ", str(self.timestamps['dropoff']), ".\n"]
        else:
            s += ["\tPending since ", str(self.time['pending']), ".\n"]

        return "".join(s)


class Simulation:
    """
    Class for containing the elements of the simulation.
    
    Attributes
    ----------
    time : int
        stores the time elapsed in the simulation
    
    num_taxis : int
        how many taxis there are
    
    request_rate : float
        rate of requests per time period
    
    hard_limit : int
        max distance from which a taxi is still assigned to a request
    
    taxis : dict
        storing all Taxi() instances in a dict
        keys are `taxi_id`s
    
    latest_taxi_id : int
        shows latest given taxi_id
        used or generating new taxis
    
    taxis_available : list of int
        stores `taxi_id`s of available taxis
    
    taxis_to_request : list of int
        stores `taxi_id`s of taxis moving to serve a request
    
    taxis_to_destination : list of int
        stores `taxi_id`s of taxis with passenger
    
    requests : dict
        storing all Request() instances in a dict
        keys are `request_id`s
    
    latest_request_id : int
        shows latest given request_id
        used or generating new requests
    
    requests_pending : list of int
        requests waiting to be served
    
    requests_in_progress : list of int
        requests with assigned taxis
    
    requests_dropped : list of int
        unsuccessful requests
    
    city : City
        geometry of class City() underlying the simulation
        
    show_map_labels : bool
    
    """

    def __init__(self, **config):

        # initializing time
        self.time = 0

        self.num_taxis = config["num_taxis"]
        self.request_rate = config["request_rate"]

        if "price_fixed" in config:
            # price that is used for every trip
            self.price_fixed = config["price_fixed"]
        else:
            self.price_fixed = 0

        if "price_per_dist" in config:
            # price per unit distance while carrying a passenger
            self.price_per_dist = config["price_per_dist"]
        else:
            self.price_per_dist = 1

        if "cost_per_unit" in config:
            # cost per unit distance (e.g. gas)
            self.cost_per_unit = config["cost_per_unit"]
        else:
            self.cost_per_unit = 0

        if "cost_per_time" in config:
            # cost per time (e.g. amortization)
            self.cost_per_time = config["cost_per_time"]
        else:
            self.cost_per_time = 0

        if "matching" in config:
            self.matching = config["matching"]

        if "batch_size" in config:
            self.batch_size = config["batch_size"]

        if "max_time" in config:
            self.max_time = config["max_time"]

        if "max_request_waiting_time" in config:
            self.max_request_waiting_time = config["max_request_waiting_time"]
        else:
            self.max_request_waiting_time = 10000 #TODO

        if ("batch_size" in config) and ("max_time" in config):
            self.num_iter = int(np.ceil(self.max_time/self.batch_size))
        else:
            self.num_iter = None

        self.taxis = RandomDict()
        self.latest_taxi_id = 0

        self.taxis_available = RandomDict()
        self.taxis_to_request = set()
        self.taxis_to_destination = set()

        self.requests = RandomDict()
        self.latest_request_id = 0

        self.requests_pending = set()

        # speeding up going through requests in the order of waiting times
        # they are pushed into a deque in the order of timestamps
        self.requests_pending_deque = deque()
        self.requests_pending_deque_temporary = deque()

        self.requests_in_progress = set()

        self.city = City(**config)
        # initializing a bunch of random coordinate pairs for later usage
        self.coordstack = self.city.generate_coords(length=self.max_time*self.request_rate*4)

        if "hard_limit" in config:
            self.hard_limit = config["hard_limit"]
        else:
            self.hard_limit = self.city.n+self.city.m

        self.log = config["log"]
        self.show_plot = config["show_plot"]

        # initializing simulation with taxis
        for t in range(self.num_taxis):
            self.add_taxi()

        if self.show_plot:
            #         plotting variables
            self.canvas = plt.figure()
            self.canvas_ax = self.canvas.add_subplot(1, 1, 1)
            self.canvas_ax.set_aspect('equal', 'box')
            self.cmap = plt.get_cmap('viridis')
            self.taxi_colors = np.linspace(0, 1, self.num_taxis)
            self.show_map_labels = config["show_map_labels"]
            self.show_pending = config["show_pending"]
            self.init_canvas()

    def init_canvas(self):
        """
        Initialize plot.
        
        """
        self.canvas_ax.clear()
        self.canvas_ax.set_xlim(-0.5, self.city.n - 0.5)
        self.canvas_ax.set_ylim(-0.5, self.city.m - 0.5)

        self.canvas_ax.tick_params(length=0)
        self.canvas_ax.xaxis.set_ticks(list(range(self.city.n)))
        self.canvas_ax.yaxis.set_ticks(list(range(self.city.m)))
        if not self.show_map_labels:
            self.canvas_ax.xaxis.set_ticklabels([])
            self.canvas_ax.yaxis.set_ticklabels([])

        self.canvas_ax.set_aspect('equal', 'box')
        self.canvas.tight_layout()
        self.canvas_ax.grid()

    def add_taxi(self):
        """
        Create new taxi.
        
        """
        # create a taxi at the base
        tx = Taxi(self.city.base_coords, self.latest_taxi_id)

        # add to taxi storage
        self.taxis[self.latest_taxi_id] = tx
        # add to available taxi storage
        self.city.A[self.city.base_coords[0], self.city.base_coords[1]].add(self.latest_taxi_id)  # add to available taxi matrix
        self.taxis_available[self.latest_taxi_id] = tx

        # increase id
        self.latest_taxi_id += 1

    def add_request(self):
        """
        Create new request.
        
        """
        # here we randomly choose a place for the request
        # the random coordinates are pre-stored in a deque for faster access
        # if there are no more pregenerated coordinates in the deque, we generate some more
        # origin
        try:
            ox, oy = self.coordstack.pop()
        except IndexError:
            self.coordstack.extend(self.city.generate_coords(length=self.max_time*self.request_rate*4))
            ox, oy = self.coordstack.pop()

        # destination
        try:
            dx, dy = self.coordstack.pop()
        except IndexError:
            self.coordstack.extend(self.city.generate_coords(length=self.max_time*self.request_rate*4))
            dx, dy = self.coordstack.pop()

        r = Request([ox, oy], [dx, dy], self.latest_request_id, self.time)

        # add to request storage
        self.requests[self.latest_request_id] = r
        # add to free users
        self.requests_pending_deque.appendleft(self.latest_request_id)
        self.requests_pending.add(self.latest_request_id)

        self.latest_request_id += 1

    def go_to_base(self, taxi_id, bcoords):
        """
        This function sends the taxi to the base rom wherever it is.
        """
        # actual coordinates
        acoords = [self.taxis[taxi_id].x, self.taxis[taxi_id].y]
        # path between actual coordinates and destination
        path = self.city.create_path(acoords, bcoords)

        # fetch object
        t = self.taxis[taxi_id]
        # erase path memory
        t.with_passenger = False
        t.to_request = False
        t.available = True
        #        print("Erasing path memory, Taxi "+str(taxi_id)+".")
        t.next_destination = Queue()
        # put path into taxi path queue
        #        print("Filling path memory, Taxi "+str(taxi_id)+". Path ",path)
        for p in path:
            t.next_destination.put(p)

        # put object back to its place
        self.taxis[taxi_id]=t

    def assign_request(self, request_id, taxi_id):
        """
        Given a request_id, taxi_id pair, this function makes the match.
        It sets new state variables for the request and the taxi, updates path of the taxi etc.
        """
        r = self.requests[request_id]
        t = self.taxis[taxi_id]

        # pair the match
        t.actual_request_executing = request_id
        r.taxi_id = taxi_id

        # remove taxi from the available ones
        self.city.A[t.x, t.y].remove(taxi_id)
        del self.taxis_available[taxi_id]
        t.with_passenger = False
        t.available = False
        t.to_request = True

        # mark taxi as moving to request
        self.taxis_to_request.add(taxi_id)

        # forget the path that has been assigned
        t.next_destination = Queue()

        # create new path: to user, then to destination
        path = self.city.create_path([t.x, t.y], [r.ox, r.oy]) + \
           self.city.create_path([r.ox, r.oy], [r.dx, r.dy])[1:]
        for p in path:
            t.next_destination.put(p)

        # remove request from the pending ones, label it as "in progress"
        self.requests_pending.remove(request_id)
        self.requests_in_progress.add(request_id)
        r.mode = 'waiting'

        # update taxi state in taxi storage
        self.taxis[taxi_id] = t
        # update request state
        self.requests[request_id] = r

        if self.log:
            print("\tM request " + str(request_id) + " taxi " + str(taxi_id))

    def check_waiting_times(self):
        """
        At the end of each assignment turn, this function drops requests that have been waiting for too long.

        """

        # making local_variables
        right = self.requests_pending_deque_temporary
        left = self.requests_pending_deque
        max_time = self.max_request_waiting_time


        # if there is something in the temporary list
        if len(right)>0:
            # if the temporary list starts with a too old element
            if self.requests[right[0]].time['pending'] >= max_time:
                # clear temporary list
                right.clear()
                # prune original list
                for i in range(len(left)):
                    if i>0 and self.requests[left[-i-1]].time['pending'] >= max_time:
                        left.pop()
                    else:
                        break
            else:
                # prune temporary list
                for i in range(len(right)):
                    if i>0 and self.requests[right[-i-1]].time['pending'] >= max_time:
                        right.pop()
                    else:
                        break
        else:
            # prune original list
            for i in range(len(left)):
                if i>0 and self.requests[left[-i - 1]].time['pending'] >= max_time:
                    left.pop()
                else:
                    break

        left.extend(right)
        self.requests_pending_deque_temporary.clear()
        self.requests_pending_deque = left
        self.requests_pending=set(left)

    def matching_algorithm(self, mode="baseline"):
        """
        This function contains the possible matching functions which are selected by the mode keyword.

        Parameters
        ----------

        mode : str, default baseline
            matching algorithm mode
                * baseline : assigning a random taxi to the user
                * request_optimized : least possible waiting times or users
                * distance_based_match : sending the nearest available taxi for the user
                * levelling : based on different measures, we want to equalize payment for taxi drivers
        """

        if len(self.requests_pending)==0:
            if self.log:
                print("No pending requests.")
            return

        if mode == "baseline_random_user_random_taxi":

            while len(self.requests_pending_deque) != 0 and len(self.taxis_available) != 0:
                # select a random taxi
                taxi_id = self.taxis_available.random_key()

                # select oldest request from deque
                request_id = self.requests_pending_deque.pop()

                # make assignment
                self.assign_request(request_id, taxi_id)

            self.check_waiting_times()

        elif mode == "baseline_random_user_nearest_taxi":

            while len(self.requests_pending_deque) != 0 and len(self.taxis_available) != 0:

                # select oldest request from deque
                request_id = self.requests_pending_deque.pop()

                # fetch request
                r = self.requests[request_id]
                # search for nearest free taxis
                possible_taxi_ids = self.find_nearest_available_taxis([r.ox, r.oy])

                # if there were any taxis near
                if len(possible_taxi_ids) > 0:
                    # select taxi
                    taxi_id = choice(possible_taxi_ids)
                    self.assign_request(request_id, taxi_id)
                else:
                    # mark request as still pending
                    self.requests_pending_deque_temporary.appendleft(request_id)

            self.check_waiting_times()

        elif mode == "levelling2_random_user_nearest_poorest_taxi_w_waiting_limit":
            # always order taxi that has earned the least money so far
            # but choose only from the nearest ones
            # hard limiting: e.g. if there is no taxi within the radius, then quit

            # go through the pending requests in a random order
            # rp_list = deepcopy(self.requests_pending)
            # shuffle(rp_list)

            # evaulate the earnings of the available taxis so far
            ta_list = list(self.taxis_available.keys)
            taxi_earnings = [self.eval_taxi_income(taxi_id) for taxi_id in ta_list]
            ta_list = list(np.array(ta_list)[np.argsort(taxi_earnings)])

            while len(self.requests_pending_deque) != 0 and len(self.taxis_available) != 0:

                # select oldest request from deque
                request_id = self.requests_pending_deque.pop()

                # fetch request
                r = self.requests[request_id]

                # find nearest vehicles in a radius
                possible_taxi_ids = self.find_nearest_available_taxis([r.ox,r.oy],mode="circle",radius=self.hard_limit)

                hit = 0
                for t in ta_list:
                    if t in possible_taxi_ids:
                        # on first hit
                        # make assignment
                        self.assign_request(request_id, t)
                        hit=1
                        break
                if not hit:
                    self.requests_pending_deque_temporary.appendleft(request_id)

            self.check_waiting_times()

        else:
            print("I know of no such assigment mode! Please provide a valid one!")

    def pickup_request(self, request_id):
        """
        Pick up passenger.
        
        Parameters
        ----------
        
        request_id : int
        """

        # mark pickup timestamp
        r = self.requests[request_id]
        t = self.taxis[r.taxi_id]

        self.taxis_to_request.remove(r.taxi_id)
        self.taxis_to_destination.add(r.taxi_id)

        # change taxi state to with passenger
        t.to_request = False
        t.with_passenger = True
        t.available = False

        r.timestamps['pickup'] = self.time
        r.mode='serving'

        # update request and taxi instances
        self.requests[request_id] = r
        self.taxis[r.taxi_id] = t
        if self.log:
            print('\tP ' + "request " + str(request_id) + ' taxi ' + str(t.taxi_id))

    def dropoff_request(self, request_id):
        """
        Drop off passenger, when taxi reached request destination.
        
        """

        # mark dropoff timestamp
        r = self.requests[request_id]
        r.timestamps['dropoff'] = self.time
        r.mode = 'done'
        t = self.taxis[r.taxi_id]

        # change taxi state to available
        t.with_passenger = False
        t.available = True
        t.actual_request_executing = None

        # update taxi lists
        self.city.A[t.x, t.y].add(r.taxi_id)
        self.taxis_to_destination.remove(r.taxi_id)
        self.taxis_available[r.taxi_id] = t

        # update request lists
        self.requests_in_progress.remove(request_id)
        t.requests_completed.add(request_id)

        # update request and taxi instances
        self.requests[request_id] = r
        self.taxis[r.taxi_id] = t
        if self.log:
            print("\tD request " + str(request_id) + ' taxi ' + str(t.taxi_id))

    def find_nearest_available_taxis(
            self,
            source,
            mode="nearest",
            radius=None):
        """
        This function lists the available taxis according to mode.

        Parameters
        ----------

        source : tuple, no default
            coordinates of the place from which we want to determine the nearest
            possible taxi

        mode : str, default "nearest"
            determines the mode of taxi listing
                * "nearest" lists only the nearest taxis, returns a list where there \
                are all taxis at the nearest possible distance from the source

                * "circle" lists all taxis within a certain distance of the source

        radius : int, optional
            if mode is "circle", gives the circle radius
        """
        frontier = [source]
        visited = []
        possible_plate_numbers = []

        distance = 0
        taxis_found=0
        while distance <= self.hard_limit and taxis_found<=len(self.taxis_available):
            # check available taxis in given nodes
            for x, y in list(frontier):
                visited.append((x, y))  # mark the coordinate as visited
                for t in self.city.A[x, y]:  # get all available taxis there
                    possible_plate_numbers.append(t)
                    taxis_found+=1
            # if we visited everybody, break
            if len(visited) == self.city.n * self.city.m:
                break
            # if we got available taxis in nearest mode, break
            if (mode == "nearest") and (len(possible_plate_numbers) > 0):
                break
            # if we reached our desired depth, break
            if (mode == "circle") and (distance > radius):
                break
            # if not, move on to next depth
            else:
                new_frontier = set()
                for f in frontier:
                    new_frontier = \
                        new_frontier.union(self.city.N[f[0],f[1]]).difference(set(visited))
                frontier = list(new_frontier)
                distance += 1

        return possible_plate_numbers

    def eval_taxi_income(self,taxi_id):
        """

        Parameters
        ----------
        taxi_id : int
            select taxi from self.taxis with id

        Returns
        -------
        price : int
            evaulated earnnigs of the taxi based on config

        """

        t = self.taxis[taxi_id]

        price =\
            self.price_fixed +\
            t.time_serving*self.price_per_dist -\
            (t.time_cruising+t.time_serving+t.time_to_request)*self.cost_per_unit -\
            (t.time_serving+t.time_cruising+t.time_to_request+t.time_waiting)*self.cost_per_time

        return price

    def plot_simulation(self):
        """
        Draws current state of the simulation on the predefined grid of the class.
        
        Is based on the taxis and requests and their internal states.
        """

        self.init_canvas()

        for taxi_id,i in self.taxis.keys.items():#TODO nem biztos, hogy működik
            t = self.taxis[taxi_id]

            # plot a circle at the place of the taxi
            self.canvas_ax.plot(t.x, t.y, 'o', ms=10, c=self.cmap(self.taxi_colors[i]))

            if self.show_map_labels:
                self.canvas_ax.annotate(
                    str(i),
                    xy=(t.x, t.y),
                    xytext=(t.x, t.y),
                    ha='center',
                    va='center',
                    color='white'
                )

            # if the taxi has a path ahead of it, plot it
            if t.next_destination.qsize() != 0:
                path = np.array([[t.x, t.y]] + list(t.next_destination.queue))
                if len(path) > 1:
                    xp, yp = path.T
                    # plot path
                    self.canvas_ax.plot(
                        xp,
                        yp,
                        '-',
                        c=self.cmap(self.taxi_colors[i])
                    )
                    # plot a star at taxi destination
                    self.canvas_ax.plot(
                        path[-1][0],
                        path[-1][1],
                        '*',
                        ms=5,
                        c=self.cmap(self.taxi_colors[i])
                    )

            # if a taxi serves a request, put request on the map
            request_id = t.actual_request_executing
            if (request_id is not None) and (not t.with_passenger):
                r = self.requests[request_id]
                self.canvas_ax.plot(
                    r.ox,
                    r.oy,
                    'ro',
                    ms=3
                )
                if self.show_map_labels:
                    self.canvas_ax.annotate(
                        request_id,
                        xy=(r.ox, r.oy),
                        xytext=(r.ox - 0.2, r.oy - 0.2),
                        ha='center',
                        va='center'
                    )

        # plot taxi base
        self.canvas_ax.plot(
            self.city.base_coords[0],
            self.city.base_coords[1],
            'ks',
            ms=15
        )

        # plot pending requests
        if self.show_pending:
            for request_id in self.requests_pending:
                self.canvas_ax.plot(
                    self.requests[request_id].ox,
                    self.requests[request_id].oy,
                    'ro',
                    ms=3,
                    alpha=0.5
                )

        self.canvas.show()

    def move_taxi(self, taxi_id):
        """
        Move a taxi one step forward according to its path queue.
        
        Update taxi position on availablity grid, if necessary.
        
        Parameters
        ----------
        plt.ion()
        taxi_id : int
            unique id of taxi that we want to move
        """
        t = self.taxis[taxi_id]

        try:
            # move taxi one step forward    
            move = t.next_destination.get_nowait()

            old_x = t.x
            old_y = t.y

            t.x = move[0]
            t.y = move[1]

            if t.with_passenger:
                t.time_serving += 1
            else:
                if t.available:
                    t.time_cruising += 1
                else:
                    t.time_to_request +=1

            # move available taxis on availability grid
            if t.available:
                self.city.A[old_x, old_y].remove(taxi_id)
                self.city.A[t.x, t.y].add(taxi_id)

            if self.log:
                print("\tF moved taxi " + str(taxi_id) + " remaining path ", list(t.next_destination.queue), "\n",
                      end="")
        except:
            t.time_waiting += 1

        self.taxis[taxi_id] = t

    def run_batch(self, run_id):
        """
        Create a batch run, where metrics are evaluated at every batch step and at the end.
        
        Parameters
        ----------
        
        run_id : str
            id that stands for simulation
        """

        measurement = Measurements(self)

        if self.num_iter is None:
            print("No batch run parameters were defined in the config file, please add them!")
            return

        print("Running simulation with run_id "+run_id+".")
        print("Batch time "+str(self.batch_size)+".")
        print("Number of iterations "+str(self.num_iter)+".")
        print("Total time simulated "+str(self.batch_size*self.num_iter)+".")
        print("Starting...")

        data_path = 'results'

        t = []
        w = []
        fields = set()
        results = []

        time1 = time()
        for i in range(self.num_iter):
            # tick the clock
            for k in range(self.batch_size):
                self.step_time("")

            ptm = measurement.read_per_taxi_metrics()
            prm = measurement.read_per_request_metrics()
            results.append(measurement.read_aggregated_metrics(ptm,prm))
            time2=time()
            print('Simulation batch '+str(i+1)+'/'+str(self.num_iter)+' , %.2f sec/batch.' % (time2-time1))
            time1=time2

        # dumping batch results
        pd.DataFrame.from_dict(results).to_csv(data_path + '/run_' + run_id + '_aggregates.csv')

        # dumping per taxi metrics out
        f = open(data_path + '/run_' + run_id + '_per_taxi_metrics.json', 'w')
        json.dump(ptm,f)
        f.close()

        # dumping per request metrics out
        f = open(data_path + '/run_' + run_id + '_per_request_metrics.json', 'w')
        json.dump(prm, f)
        f.close()

        print("Done.\n")


    def step_time(self, handler):
        """
        Ticks simulation time by 1.
        """

        if self.log:
            print("timestamp " + str(self.time))

        # move every taxi one step towards its destination
        for taxi_id in self.taxis:
            self.move_taxi(taxi_id)

            t = self.taxis[taxi_id]

            # if a taxi can pick up its passenger, do it
            if taxi_id in self.taxis_to_request:
                r = self.requests[t.actual_request_executing]
                if (t.x == r.ox) and (t.y == r.oy):
                    self.pickup_request(t.actual_request_executing)
            # if a taxi can drop off its passenger, do it
            elif taxi_id in self.taxis_to_destination:
                r = self.requests[t.actual_request_executing]
                if (t.x == r.dx) and (t.y == r.dy):
                    self.dropoff_request(r.request_id)
                    self.go_to_base(taxi_id, self.city.base_coords)

        # generate requests
        for i in range(self.request_rate):
            self.add_request()

        # make matchings
        self.matching_algorithm(mode=self.matching)

        # update timer inside requests
        for request_id in self.requests:
            self.requests[request_id].timer()

        # step time
        if self.show_plot:
            self.plot_simulation()
        self.time += 1

class Measurements:

    def __init__(self,simulation):
        self.simulation = simulation

    def read_per_taxi_metrics(self):
        """
        Returns metrics for taxis.

        Outputs a dictionary that stores these metrics in lists and the timestamp of the call.

        Output
        ------

        timestamp: int
            the timestamp of the measurement

        avg_trip_length: list of floats
            average trip length

        std_trip_length: list of floats
            standard deviation of trip lengths per taxi

        avg_trip_price: list of floats
            average trip price per taxi

        std_trip_price: list of floats
            standard deviation of trip price per taxi

        number_of_requests_completed: list of ints
            how many passengers has the taxi served

        ratio_online: list of floats
            ratio of useful travel time from overall time per taxi
            online/(online+to_request+cruising+waiting)

        ratio_to_request: list of floats
            ratio of empty travel time (from assignment to pickup)

        ratio_cruising: list of floats
            ratio of travelling with no assigned request

        ratio_waiting: list of floats
            ratio of standing at a post with no assigned request
        """

        # for the taxis

        # average trip lengths per taxi
        # standard deviation of trip lengths per taxi
        trip_avg_length = []
        trip_std_length = []
        trip_avg_price = []
        trip_num_completed = []

        ratio_online = []
        ratio_serving = []
        ratio_to_request = []
        ratio_cruising = []
        ratio_waiting = []

        for taxi_id in self.simulation.taxis:
            taxi = self.simulation.taxis[taxi_id]
            req_lengths = []
            req_prices = []

            for request_id in taxi.requests_completed:
                r = self.simulation.requests[request_id]
                length = np.abs(r.dy - r.oy) + np.abs(r.dx - r.ox)
                req_lengths.append(length)

            if len(req_lengths) > 0:
                trip_avg_length.append(np.nanmean(req_lengths))
                trip_std_length.append(np.nanstd(req_lengths))
            else:
                trip_avg_length.append(0)
                trip_std_length.append(np.nan)
            trip_num_completed.append(len(req_prices))
            trip_avg_price.append(self.simulation.eval_taxi_income(taxi_id))

            s = taxi.time_serving
            w = taxi.time_waiting
            r = taxi.time_to_request
            c = taxi.time_cruising
            total = s+w+r+c

            ratio_serving.append( s / total )
            ratio_cruising.append( c / total )
            ratio_online.append( (s+r) / total )
            ratio_waiting.append( w / total )
            ratio_to_request.append( r / total )

        return {
            "timestamp": self.simulation.time,
            "trip_avg_length": trip_avg_length,
            "trip_std_length": trip_std_length,
            "trip_avg_price": trip_avg_price,
            "trip_num_completed": trip_num_completed,
            "ratio_serving": ratio_serving,
            "ratio_cruising": ratio_cruising,
            "ratio_online":ratio_online,
            "ratio_waiting":ratio_waiting,
            "ratio_to_request":ratio_to_request
        }

    def read_per_request_metrics(self):

        # for the requests

        request_completed = []
        total = 0
        request_last_waiting_times = []
        request_lengths = []
        dropped_coords = []

        for request_id in self.simulation.requests:
            r = self.simulation.requests[request_id]
            if r.mode == 'done':
                request_completed.append(1)
                request_lengths.append(float(np.abs(r.dy - r.oy) + np.abs(r.dx - r.ox)))
            elif r.mode == 'dropped':
                request_completed.append(0)
                dropped_coords.append([r.ox,r.oy])
            else:
                request_completed.append(0)

            total += 1
            # to forget history
            # system-level waiting time peak detection
            # it would not be sensible to include all previous waiting times
            if (self.simulation.time - r.timestamps['request']) < 100:
                if r.mode == 'pending':
                    request_last_waiting_times.append(r.time['pending'])

        return {
            "timestamp" : self.simulation.time,
            "request_completed" : request_completed,
            "request_last_waiting_times" : request_last_waiting_times,
            "request_lengths" : request_lengths,
            "dropped_coords" : dropped_coords
        }

    @staticmethod
    def read_aggregated_metrics(per_taxi_metrics, per_request_metrics):
        metrics = {"timestamp": per_taxi_metrics["timestamp"]}

        for k in per_taxi_metrics:
            if k[0:4]=='trip':
                if k[5]=='a':
                    metrics['avg_'+k] = np.nanmean(per_taxi_metrics[k])
                    metrics['std_' + k] = np.nanstd(per_taxi_metrics[k])
            elif k[0:5]=='ratio':
                metrics['avg_' + k] = np.nanmean(per_taxi_metrics[k])
                metrics['std_' + k] = np.nanstd(per_taxi_metrics[k])
                y,x = np.histogram(per_taxi_metrics[k],bins=1000,density=True)
                metrics['entropy_' + k] = entropy(y)

        for k in per_request_metrics:
            if k[0] != 'd' or k[0] != 't':
                metrics['avg_' + k] = np.nanmean(per_request_metrics[k])
                metrics['std_' + k] = np.nanstd(per_request_metrics[k])

        return metrics