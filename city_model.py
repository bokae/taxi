#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:00:15 2017

@author: bokanyie
"""

import numpy as np
import pandas as pd
import json
from random import choice, shuffle
import matplotlib.pyplot as plt
from time import time

import gzip
import shutil
import os

# special data types
from collections import deque
from queue import Queue
from randomdict import RandomDict

from geometry import City


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

   home : tuple
        if config setting is "initial_conditions":"random", then this will be the home of the taxi instead of the base

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
            self.next_destination = deque()  # path to travel

            # this can only be filled if the city geometry is known
            self.home = None

    def __str__(self):
        """
        Verbose string representation of taxi for debugging purposes.

        Returns
        -------
        string

        """
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

    def __iter__(self):
        """
        This method converts the class into a dict, with attributes as keys.
        """
        for attr, value in self.__dict__.items():
            yield attr, value


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
    
    taxi_id : int
        id of taxi that serves the request

    mode : str
        current mode

    timestamps : dict
        stores simulation timestamps of the change of mode
    """

    def __init__(self, ocoords=None, dcoords=None, request_id=None, timestamp=None):
        """
        
        """
        if (ocoords is None) or (dcoords is None):
            print("A request has to have a well-defined origin and destination.")
        elif request_id is None:
            print("Please identify each request uniquely.")
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
                'request': timestamp,
                'assigned': None,
                'pickup': None,
                'dropoff': None
            }

            self.mode = 'pending'

    def __str__(self):
        """
        Verbose string representation of request for debugging purposes.

        Returns
        -------
        string

        """
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
            s += ["\tWaiting since ", str(self.timestamp['request']), ".\n"]
            if self.mode != 'pending':
                s += ["\tPickup timestamp ", str(self.timestamps['pickup']), ".\n"]
                if self.timestamps['dropoff'] is not None:
                    s += ["\tDropoff timestamp ", str(self.timestamps['dropoff']), ".\n"]
        else:
            s += ["\tPending since ", str(self.time - self.timestamps['request']), ".\n"]

        return "".join(s)

    def __iter__(self):
        """
        This method converts the class into a dict, with attributes as keys.
        """
        for attr, value in self.__dict__.iteritems():
            yield attr, value


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

        # this is done now by the config generator in theory
        if "max_request_waiting_time" in config:
            self.max_request_waiting_time = config["max_request_waiting_time"]
        else:
            self.max_request_waiting_time = 10000

        if ("batch_size" in config) and ("max_time" in config):
            self.num_iter = int(np.ceil(self.max_time/self.batch_size))
        else:
            self.num_iter = None

        if "behaviour" in config: # goback / stay / cruise
            self.behaviour = config["behaviour"]
        else:
            self.behaviour = "go_back"

        if "initial_conditions" in config: # base / home
            self.initial_conditions = config["initial_conditions"]
        else:
            self.initial_conditions = "base"

        if "reset_time" in config:
            self.reset_time = config["reset_time"]
        else:
            self.reset_time = self.max_time +1

        # initializing counters
        self.latest_taxi_id = 0
        self.latest_request_id = 0

        # initializing object storage
        self.taxis = RandomDict()
        self.taxis_available = RandomDict()
        self.taxis_to_request = set()
        self.taxis_to_destination = set()

        self.requests = RandomDict()
        self.requests_pending = set()

        # speeding up going through requests in the order of waiting times
        # they are pushed into a deque in the order of timestamps
        self.requests_pending_deque = deque()
        self.requests_pending_deque_batch = deque(maxlen=self.max_request_waiting_time)
        self.requests_pending_deque_temporary = deque()
        self.requests_in_progress = set()

        # city layout
        self.city = City(**config)
        # length of pregenerated random number storage
        self.city.length = int(min(self.max_time*self.request_rate, 1e6))

        # whether to log all movements for debugging purposes
        self.log = config["log"]
        self.city.log = self.log
        # showing map of moving taxis in interactive jupyter notebook
        self.show_plot = config["show_plot"]

        # initializing simulation with taxis
        for t in range(self.num_taxis):
            self.add_taxi()

        self.taxi_df = pd.DataFrame.from_dict([dict(v) for k, v in self.taxis.items()])
        self.taxi_df.set_index('taxi_id', inplace=True)

        if self.show_plot:
            # plotting variables
            self.canvas = plt.figure()
            self.canvas_ax = self.canvas.add_subplot(1, 1, 1)
            self.canvas_ax.set_aspect('equal', 'box')
            self.cmap = plt.get_cmap('viridis')
            self.taxi_colors = list(np.linspace(0, 0.85, self.num_taxis))
            shuffle(self.taxi_colors)
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

        # adding home coordinates, starting taxi
        home = self.city.create_taxi_home_coords()

        if self.initial_conditions == "base":
            # create a taxi at the base
            tx = Taxi(self.city.base_coords, self.latest_taxi_id)
        elif self.initial_conditions == "home":
            # create a taxi at home
            tx = Taxi(home, self.latest_taxi_id)
        tx.home = home

        # add to taxi storage
        self.taxis[self.latest_taxi_id] = tx
        # add to available taxi matrix
        self.city.A[self.city.coordinate_dict_ij_to_c[tx.x][tx.y]].add(self.latest_taxi_id)
        # add to available taxi storage
        self.taxis_available[self.latest_taxi_id] = tx
        # increase counter
        self.latest_taxi_id += 1

    def add_request(self):
        """
        Create new request.
        
        """
        # here we randomly choose a place for the request
        # the random coordinates are pre-stored in a deque for faster access
        # if there are no more pregenerated coordinates in the deque, we generate some more

        # origin and destination coordinates
        ox, oy, dx, dy = self.city.create_one_request_coord()
        r = Request([ox, oy], [dx, dy], self.latest_request_id, self.time)

        # add to request storage
        self.requests[self.latest_request_id] = r
        # add to free users
        self.requests_pending_deque.append(self.latest_request_id)
        # increase counter
        self.latest_request_id += 1

    def go_to_base(self, taxi_id, bcoords):
        """
        This function sends the taxi to the base rom wherever it is.
        """

        # fetch object
        t = self.taxis[taxi_id]

        # actual coordinates
        acoords = [t.x, t.y]
        # path between actual coordinates and destination
        path = self.city.create_path(acoords, bcoords)

        # erase path memory
        t.with_passenger = False
        t.to_request = False
        t.available = True
        #        print("Erasing path memory, Taxi "+str(taxi_id)+".")
        t.next_destination = deque()
        # put path into taxi path queue
        #        print("Filling path memory, Taxi "+str(taxi_id)+". Path ",path)
        t.next_destination.extend(path)

        # put object back to its place
        self.taxis[taxi_id] = t

    def go_home_everybody(self):
        """
        Drop requests that are currently executing, and set taxis as available at their home locations.
        """
        for taxi_id in self.taxis:
            tx = self.taxis[taxi_id]

            if taxi_id in self.taxis_available:
                # (magic wand) Apparate taxi home!
                self.city.A[self.city.coordinate_dict_ij_to_c[tx.x][tx.y]].remove(taxi_id)
                self.city.A[self.city.coordinate_dict_ij_to_c[tx.home[0]][tx.home[1]]].add(taxi_id)
                tx.x, tx.y = tx.home

            if taxi_id in self.taxis_to_destination:
                # if somebody is sitting in it, finish request
                self.dropoff_request(tx.actual_request_executing, mode="going_home")

            if taxi_id in self.taxis_to_request:
                # if it was only going towards a request, cancel it
                self.dropoff_request(tx.actual_request_executing, mode="cancel")

            self.taxis[taxi_id] = tx

    def cruise(self, taxi_id):
        return None

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
        self.city.A[self.city.coordinate_dict_ij_to_c[t.x][t.y]].remove(taxi_id)
        del self.taxis_available[taxi_id]
        t.with_passenger = False
        t.available = False
        t.to_request = True

        # mark taxi as moving to request
        self.taxis_to_request.add(taxi_id)

        # forget the path that has been assigned
        t.next_destination = deque()

        # create new path: to user, then to destination
        path = self.city.create_path([t.x, t.y], [r.ox, r.oy]) + \
            self.city.create_path([r.ox, r.oy], [r.dx, r.dy])[1:]
        t.next_destination.extend(path)

        # remove request from the pending ones, label it as "in progress"
        self.requests_in_progress.add(request_id)
        r.mode = 'waiting'
        r.timestamps['assign'] = self.time

        # update taxi state in taxi storage
        self.taxis[taxi_id] = t
        # update request state
        self.requests[request_id] = r

        if self.log:
            print("\tM request " + str(request_id) + " taxi " + str(taxi_id))

    def matching_algorithm(self, mode="random_unlimited"):
        """
        This function contains the possible matching functions which are selected by the mode keyword.

        Parameters
        ----------

        mode : str, default baseline
            matching algorithm mode
                * random_unlimited : assigning a random taxi to the user
                * random_limited : assigning a random taxi to the user within the circle of a radius self.city.hard_limit
                * nearest : sending the nearest available taxi for the user from within the circle of a radius self.city.hard_limit
                * poorest : sending the least earning available taxi for the user from within the circle of a radius self.city.hard_limit
        """

        if len(self.requests_pending_deque) == 0:
            if self.log:
                print("No pending requests.")
            return

        if self.log:
            print('Matching algorithm.')

        if mode == "random_unlimited":

            while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                # select a random taxi
                taxi_id = self.taxis_available.random_key()
                # select oldest request from deque
                request_id = self.requests_pending_deque.popleft()
                # make assignment
                self.assign_request(request_id, taxi_id)

        elif mode == "random_limited":
            while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                # select oldest request from deque
                request_id = self.requests_pending_deque.popleft()
                # fetch request
                r = self.requests[request_id]
                # search for nearest free taxis
                possible_taxi_ids = self.city.find_nearest_available_taxis(
                    self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                    mode="circle",
                    radius=self.city.hard_limit
                )
                # if there were any taxis near
                if len(possible_taxi_ids) > 0:
                    # select taxi
                    taxi_id = choice(possible_taxi_ids)
                    self.assign_request(request_id, taxi_id)
                else:
                    # mark request as still pending
                    self.requests_pending_deque_temporary.append(request_id)

        elif mode == "nearest":
            while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                # select oldest request from deque
                request_id = self.requests_pending_deque.popleft()
                # fetch request
                r = self.requests[request_id]
                # search for nearest free taxis
                possible_taxi_ids = self.city.find_nearest_available_taxis(
                    self.city.coordinate_dict_ij_to_c[r.ox][r.oy])
                # if there were any taxis near
                if len(possible_taxi_ids) > 0:
                    # select taxi
                    taxi_id = choice(possible_taxi_ids)
                    self.assign_request(request_id, taxi_id)
                else:
                    # mark request as still pending
                    self.requests_pending_deque_temporary.append(request_id)

        elif mode == "poorest":
            # always order taxi that has earned the least money so far
            # but choose only from the nearest ones
            # hard limiting: e.g. if there is no taxi within the radius, then quit

            # evaulate the earnings of the available taxis so far
            ta_list = list(self.taxis_available.keys)
            # print('Available taxis ', self.taxis_available.keys)
            taxi_earnings = [self.eval_taxi_income(taxi_id) for taxi_id in ta_list]
            # print('Earnings ', taxi_earnings)
            ta_list = list(np.array(ta_list)[np.argsort(taxi_earnings)])

            pairs = 0
            while len(self.requests_pending_deque) > 0 and len(self.taxis_available) > 0:
                # select oldest request from deque
                request_id = self.requests_pending_deque.popleft()
                # fetch request
                r = self.requests[request_id]
                # find nearest vehicles in a radius
                possible_taxi_ids = self.city.find_nearest_available_taxis(self.city.coordinate_dict_ij_to_c[r.ox][r.oy],
                                                                      mode="circle",
                                                                      radius=self.city.hard_limit)
                hit = 0
                for t in ta_list:
                    if t in possible_taxi_ids:
                        # on first hit
                        # make assignment
                        self.assign_request(request_id, t)
                        hit = 1
                        pairs +=1
                        break
                if not hit:
                    self.requests_pending_deque_temporary.append(request_id)

        else:
            print("I know of no such assignment mode! Please provide a valid one!")

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
        r.mode = 'serving'

        # update request and taxi instances
        self.requests[request_id] = r
        self.taxis[r.taxi_id] = t
        if self.log:
            print('\tP ' + "request " + str(request_id) + ' taxi ' + str(t.taxi_id))

    def dropoff_request(self, request_id, mode="simple"):
        """
        Drop off passenger, when taxi reached request destination.
        
        """

        r = self.requests[request_id]
        t = self.taxis[r.taxi_id]

        if mode == "simple" or mode == "going_home":
            # mark request as done
            r.timestamps['dropoff'] = self.time
            r.mode = 'done'
            self.requests_in_progress.remove(request_id)
            t.requests_completed.add(request_id)
            # remove taxi from to_destination list
            self.taxis_to_destination.remove(r.taxi_id)
        elif mode == "cancel":
            # mark request as dropped
            r.mode = 'dropped'
            # remove request from progressing ones
            self.requests_in_progress.remove(request_id)
            # clear taxi path
            t.next_destination = deque()
            # remove taxi from to_request list
            self.taxis_to_request.remove(r.taxi_id)

        # update taxi lists
        if mode=="going_home":
            # (magic wand) Apparate taxi home!
            t.x,t.y = t.home

        # update global availability containers
        self.taxis_available[r.taxi_id] = t
        self.city.A[self.city.coordinate_dict_ij_to_c[t.x][t.y]].add(r.taxi_id)

        # update taxi internal states
        t.with_passenger = False
        t.available = True
        t.actual_request_executing = None

        # update request and taxi instances in global containers
        self.requests[request_id] = r
        self.taxis[r.taxi_id] = t

        if self.log:
            print("\tD request " + str(request_id) + ' taxi ' + str(t.taxi_id))

    def eval_taxi_income(self, taxi_id):
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
            len(t.requests_completed) * self.price_fixed +\
            int(not t.available) * self.price_fixed +\
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

        for taxi_id, i in self.taxis.keys.items():
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
            if len(t.next_destination)> 0:
                path = np.array([[t.x, t.y]] + list(t.next_destination))
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
            for request_id in self.requests_pending_deque:
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
            move = t.next_destination.popleft()

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
                    t.time_to_request += 1

            # move available taxis on availability grid
            if t.available:
                self.city.A[self.city.coordinate_dict_ij_to_c[old_x][old_y]].remove(taxi_id)
                self.city.A[self.city.coordinate_dict_ij_to_c[t.x][t.y]].add(taxi_id)
            if self.log:
                print("\tF moved taxi " + str(taxi_id) + " remaining path ", list(t.next_destination), "\n", end="")
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
        print("Number of items "+str(self.num_iter)+".")
        print("Total time simulated "+str(self.batch_size*self.num_iter)+".")
        print("Starting...")

        data_path = 'results'

        results = []

        time1 = time()
        for i in range(self.num_iter):
            # tick the clock
            for k in range(self.batch_size):
                self.step_time("")

            ptm = measurement.read_per_taxi_metrics()
            prm = measurement.read_per_request_metrics()

            if i == 0:
                # clearing future output files
                f = open(data_path + '/run_' + run_id + '_per_taxi_metrics.json', 'w')
                f.close()
                f = open(data_path + '/run_' + run_id + '_per_request_metrics.json', 'w')
                f.close()

                # adding taxi homes to output
                ptm['taxi_homes'] = [[int(self.taxis[t].home[0]), int(self.taxis[t].home[1])] for t in self.taxis]

            # dumping per taxi metrics out (per batch)
            f = open(data_path + '/run_' + run_id + '_per_taxi_metrics.json', 'a')
            json.dump(ptm, f)
            f.write('\n')
            f.close()

            # dumping per request metrics out (per batch)
            f = open(data_path + '/run_' + run_id + '_per_request_metrics.json', 'a')
            json.dump(prm, f)
            f.write('\n')
            f.close()

            results.append(measurement.read_aggregated_metrics(ptm, prm))
            time2 = time()
            print('Simulation batch '+str(i+1)+'/'+str(self.num_iter)+' , %.2f sec/batch.' % (time2-time1))

            time1 = time2

        # dumping batch results
        f = open(data_path + '/run_' + run_id + '_aggregates.csv', 'w')
        pd.DataFrame.from_dict(results).to_csv(f, float_format="%.4f")
        f.write('\n')
        f.close()

        # compressing written objects
        for file in ['_per_taxi_metrics.json', '_per_request_metrics.json', '_aggregates.csv']:
            f1 = open(data_path + '/run_' + run_id + file, 'rb')
            f2 = gzip.open(data_path + '/run_' + run_id + file + '.gz', 'wb')
            shutil.copyfileobj(f1, f2)
            f1.close()
            f2.close()
            os.remove(data_path + '/run_' + run_id + file)

        print("Done.\n")

    def step_time(self, handler):
        """
        Ticks simulation time by 1.
        """

        if self.log:
            print("\n")
            print("Timestamp " + str(self.time))
            print("Taxis:\n")
            print("\t Available: "+str(len(self.taxis_available)))
            print("\t To request: " + str(len(self.taxis_to_request)))
            print("\t To destination: " + str(len(self.taxis_to_destination)))
            print("\n")
            req_counter = {'TOTAL':self.latest_request_id}
            for request_id in self.requests:
                r = self.requests[request_id]
                if r.mode in req_counter:
                    req_counter[r.mode] += 1
                else:
                    req_counter[r.mode] = 1
            print('Requests:')
            for mode in req_counter:
                print('\t'+mode+': '+str(req_counter[mode]))
            print("\n")
            print("Requests pending: ")
            print(self.requests_pending)
            print(self.requests_pending_deque)
            print(self.requests_pending_deque_temporary)

        if (self.time > 0) and (self.time % self.reset_time == 0):
            # print("Going home!")
            self.go_home_everybody()
        else:
            # move every taxi one step towards its destination
            for taxi_id in self.taxis:
                self.move_taxi(taxi_id)

                t = self.taxis[taxi_id]

                # if a taxi can pick up its passenger, do it
                if taxi_id in self.taxis_to_request:
                    r = self.requests[t.actual_request_executing]
                    if (t.x == r.ox) and (t.y == r.oy):
                        try:
                            self.pickup_request(t.actual_request_executing)
                        except KeyError:
                            print(t)
                            print(r)
                # if a taxi can drop off its passenger, do it
                elif taxi_id in self.taxis_to_destination:
                    r = self.requests[t.actual_request_executing]
                    if (t.x == r.dx) and (t.y == r.dy):
                        self.dropoff_request(r.request_id)
                        if self.behaviour == "go_back":
                            if self.initial_conditions == "base":
                                self.go_to_base(taxi_id, self.city.base_coords)
                            elif self.initial_conditions == "home":
                                self.go_to_base(taxi_id, t.home)
                        elif self.behaviour == "stay":
                            pass
                        elif self.behaviour == "cruise":
                            self.cruise(taxi_id)
        # make matchings
        self.matching_algorithm(mode=self.matching)

        # reunite pending requests
        self.requests_pending_deque_temporary.reverse()
        self.requests_pending_deque.extendleft(self.requests_pending_deque_temporary)
        self.requests_pending_deque_temporary = deque()
        # delete old requests from pending ones
        if self.time > self.max_request_waiting_time and len(self.requests_pending_deque) > 0:
            while len(self.requests_pending_deque)>0 and (self.requests_pending_deque[0] in self.requests_pending_deque_batch[0]):
                request_id = self.requests_pending_deque.popleft()
                self.requests[request_id].mode = 'dropped'

        self.requests_pending = set(self.requests_pending_deque)

        # generate requests
        new_requests = set()
        rfrac, rint = np.modf(self.request_rate)
        for i in range(int(rint)):
            self.add_request()
            new_requests.add(self.latest_request_id)
        if rfrac > 1e-3:
            try:
                p = self.city.request_p.pop()
            except IndexError:
                self.city.request_p.extend(np.random.random(self.city.length))
                p = self.city.request_p.pop()
            if p < rfrac:
                self.add_request()
                new_requests.add(self.latest_request_id)

        # this automatically pushes out requests that have been waiting for too long
        self.requests_pending_deque_batch.append(new_requests)

        if self.show_plot:
            self.plot_simulation()

        # step time
        self.time += 1


class Measurements:

    def __init__(self, simulation):
        self.simulation = simulation

    def read_per_taxi_metrics(self):
        """
        Returns metrics for taxis.

        Outputs a dictionary that stores these metrics in lists and the timestamp of the call.

        Output
        ------

        timestamp: int
            the timestamp of the measurement

        trip_avg_length: list of floats
            average trip lengths per taxi

        trip_std_length: list of floats
            standard deviation of trip lengths per taxi

        trip_income: list of floats
            average trip income per taxi

        trip_num_completed: int
            number of trips completed by the taxi

        time_serving: list of floats
            time of useful travel time from overall time per taxi

        time_to_request: list of floats
            time of empty travel time (from assignment to pickup)

        time_cruising: list of floats
            time of travelling with no assigned request

        time_waiting: list of floats
            time of standing in place with no assigned request

        position: (int,int)
            current taxi position on grid
        """

        # for the taxis

        # average trip lengths per taxi
        trip_avg_length = []
        trip_std_length = []
        incomes = []
        trip_num_completed = []
        
        time_serving = []
        time_to_request = []
        time_cruising = []
        time_waiting = []
        
        position = []

        for taxi_id in self.simulation.taxis:
            taxi = self.simulation.taxis[taxi_id]
            req_lengths = []

            for request_id in taxi.requests_completed:
                r = self.simulation.requests[request_id]
                length = float(np.abs(r.dy - r.oy) + np.abs(r.dx - r.ox))
                req_lengths.append(length)

            if len(req_lengths) > 0:
                trip_avg_length.append( round(np.nanmean(req_lengths), 4) )
                trip_std_length.append( round(np.nanstd(req_lengths), 4) )
            else:
                trip_avg_length.append(0)
                trip_std_length.append(np.nan)
            trip_num_completed.append(len(req_lengths))
            incomes.append(self.simulation.eval_taxi_income(taxi_id))

            s = taxi.time_serving
            w = taxi.time_waiting
            r = taxi.time_to_request
            c = taxi.time_cruising
            
            time_serving.append(s)
            time_cruising.append(c)
            time_waiting.append(w)
            time_to_request.append(r)

            position.append([int(taxi.x), int(taxi.y)])

        return {
            "timestamp": self.simulation.time,
            "trip_avg_length": trip_avg_length,
            "trip_std_length": trip_std_length,
            "trip_income": incomes,
            "trip_num_completed": trip_num_completed,
            "time_serving": time_serving,
            "time_cruising": time_cruising,
            "time_waiting": time_waiting,
            "time_to_request": time_to_request,
            "position": position
        }

    def read_per_request_metrics(self):
        """
        Returns aggregated metrics for requests.

        Outputs a dictionary that stores these metrics in lists and the timestamp of the call.

        Output
        -------

        timestamp: int
            the timestamp of the measurement

        request_completed:
            fraction of finished requests from the overall pool

        request_last_waiting_times:
            average waiting times for latest requests from the pool

        request_lengths:
            average lengths of completed requests

        """

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
                dropped_coords.append([r.ox, r.oy])
            else:
                request_completed.append(0)

            total += 1
            # to forget history
            # system-level waiting time peak detection
            # it would not be sensible to include all previous waiting times
            if (self.simulation.time - r.timestamps['request']) < 100:
                if r.mode == 'pending':
                    request_last_waiting_times.append(self.simulation.time - r.timestamps['request'])

        request_waiting_time_distribution = \
            {
                (self.simulation.max_request_waiting_time-i-1):
                    len(self.simulation.requests_pending.intersection(l))
                    for i, l in enumerate(self.simulation.requests_pending_deque_batch)
            }

        return {
            "timestamp": self.simulation.time,
            "request_completed": round(np.mean(request_completed), 4),
            "request_last_waiting_times": round(np.mean(request_last_waiting_times), 4),
            "request_lengths": round(np.mean(request_lengths), 4),
            "requests_waiting_time_distr": request_waiting_time_distribution
        }

    @staticmethod
    def read_aggregated_metrics(per_taxi_metrics, per_request_metrics):

        metrics = {"timestamp": per_taxi_metrics["timestamp"]}

        for k in per_taxi_metrics:
            if k[0:6] == 'trip_i':
                metrics['avg_'+k] = np.nanmean(per_taxi_metrics[k])
                metrics['std_' + k] = np.nanstd(per_taxi_metrics[k])
            elif k[0:4] == 'time':
                metrics['avg_' + k] = np.nanmean(per_taxi_metrics[k])
                metrics['std_' + k] = np.nanstd(per_taxi_metrics[k])

        return metrics
    