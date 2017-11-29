#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:00:15 2017

@author: bokanyie
"""

import numpy as np
from random import shuffle,choice
import matplotlib.pyplot as plt
from queue import Queue

class City():
    """
    Represents a grid on which taxis are moving.
    """

    def __init__(self,**config):
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
        if (("n" in config) and ("m" in config)):
            self.n = config["n"] # number of pixels in x direction
            self.m = config["m"] # number of pixels in y direction

            # array that stores taxi_id of available taxis at the
            # specific position on the grid
            # we initialize this array with empy lists
            self.A = np.empty((self.n,self.m),dtype=list)
            for i in range(self.n):
                for j in range(self.m):
                    self.A[i,j]=list()
                    
            self.base_coords = [int(np.floor(self.n/2)),int(np.floor(self.m/2))]
#            print(self.base_coords)
            self.base_sigma = config["base_sigma"]

    def measure_distance(self,source,destination):
        """
        Measure distance on the grid between two points.
        
        Returns
        -------
        Source coordinates are marked by *s*,
        destination coordinates are marked by *d*.
        
        The distance is the following integer:
        $$|x_s-x_d|+|y_s-y_d|$$
        """

        return np.dot(np.abs(np.array(destination)-np.array(source)),[1,1])

    def create_path(self,source,destination):
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
        dx, dy = np.array(destination) - np.array(source)
        # create a sequence of "x"-es and "y"-s
        # we are going to shuffle this sequence
        # to get a random order of "x" and "y" direction steps
        sequence = ['x']*int(np.abs(dx))+['y']*int(np.abs(dy))
        shuffle(sequence)
        # source is included in the path
        path=[source]
        for item in sequence:
            if item=="x":
                # we add one step in the "x" direction based on the last position
                path.append([np.sign(dx)+path[-1][0],0+path[-1][1]])
            else:
                # we add one step in the "y" direction based on the last position
                path.append([0+path[-1][0],np.sign(dy)+path[-1][1]])
        return path

    def neighbors(self,coordinates):
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

        ns = set()
        for dx,dy in [(1,0),(0,1),(-1,0),(0,-1)]:
            new_x = coordinates[0]+dx
            new_y = coordinates[1]+dy

            if ((0<=new_x) and (self.n>new_x) and (0<=new_y) and (self.m>new_y)):
                ns.add((new_x,new_y))
        return ns
    
    def create_request_coords(self):
        """
        Creates random request coords based on the base_coords and the
        base_sigma according to a 2D Gauss.
        
        Returns
        -------
        
        x,y
            coordinate tuple
        """
        
        done=False
        while(not done):
            mean = self.base_coords
            cov = np.array([[self.base_sigma**2,0],[0,self.base_sigma**2]])
            
            x,y=np.random.multivariate_normal(mean,cov)
            
            x = int(np.floor(x))
            y = int(np.floor(y))
            
            if ((x>=0) and (x<self.n) and (y>=0) and (y<self.m)):
                done=True
        return x,y

class Taxi():
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
        
    waiting_time : int
        time spent with empty waiting
    
    useful_travel_time : int
        time spent with carrying a passenger
        
    empty_travel_time : int
        time spent with travelling to a request
        
    next_destination : Queue
        Queue that stores the path forward of the taxi
    
    """
    def __init__(self,coords = None,taxi_id = None):
        if coords == None:
            print("You have to put your taxi somewhere in the city!")
        elif taxi_id == None:
            print("Not a licenced taxi.")
        else:
            
            self.x=coords[0]
            self.y=coords[1]
            
            self.taxi_id=taxi_id
            
            self.available=True
            self.to_request=False
            self.with_passenger=False
            
            self.actual_request_executing=None
            self.requests_completed=[]
                        
            # metrics
            self.waiting_time=0
            self.useful_travel_time=0
            self.empty_travel_time=0

            # storing steps to take
            self.next_destination=Queue() # path to travel

class Request():
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

    def __init__(self,ocoords = None, dcoords=None, request_id = None, timestamp = None):
        """
        
        """
        if (ocoords == None) or (dcoords==None):
            print("A request has to have a well-defined origin and destination.")
        elif request_id == None:
            print("Please indentify each request uniquely.")
        elif timestamp == None:
            print("Please give a timestamp for the request!")
        else:
            # pickup coordinates
            self.ox = ocoords[0]
            self.oy = ocoords[1]
            
            # desired dropoff coordinates
            self.dx = dcoords[0]
            self.dy = dcoords[1]

            self.request_id = request_id

            self.request_timestamp = timestamp
            
            # travel data
            self.pickup_timestamp = None
            self.dropoff_timestamp = None
            self.taxi_id=None

            self.waiting_time=0
            
            

class Simulation():
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
    
    requests_fulfilled : list of int
        requests that are closed
    
    requests_dropped : list of int
        unsuccessful requests
    
    city : City
        geometry of class City() underlying the simulation
    
    """

    def __init__(self,**config):
        
        # initializing empty grid
        self.time = 0

        self.num_taxis = config["num_taxis"]
        self.request_rate = config["request_rate"]
        self.hard_limit = config["hard_limit"]

        self.taxis={}
        self.latest_taxi_id = 0
        
        self.taxis_available=[]
        self.taxis_to_request=[]
        self.taxis_to_destination=[]

        self.requests={}
        self.latest_request_id = 0
        
        self.requests_pending=[]
        self.requests_in_progress=[]
        self.requests_fulfilled=[]
        self.requests_dropped=[]

        self.city = City(**config)

        # initializing simulation with taxis
        for t in range(self.num_taxis):
            self.add_taxi(t)

        # plotting variables
#        self.canvas = plt.figure()
#        self.canvas_ax = self.canvas.add_subplot(1,1,1)
#        self.cmap = plt.get_cmap('viridis')
#        self.taxi_colors = np.random.rand(self.num_taxis)
#        self.init_canvas()

    def init_canvas(self):
        """
        Initialize plot.
        
        """
        self.canvas_ax.clear()
        self.canvas_ax.set_xlim(-0.5,self.city.n-0.5)
        self.canvas_ax.set_ylim(-0.5,self.city.m-0.5)

        self.canvas_ax.xaxis.set_ticks(list(range(self.city.n)))
        self.canvas_ax.yaxis.set_ticks(list(range(self.city.m)))

        self.canvas_ax.grid()

    def add_taxi(self,taxi_id):
        """
        Create new taxi.
        
        """
        # create a taxi at the base
        tx = Taxi(self.city.base_coords,self.latest_taxi_id)
        
        # add to taxi storage
        self.taxis[self.latest_taxi_id]= tx
        # add to available taxi storage
        self.city.A[self.city.base_coords[0],self.city.base_coords[1]].append(self.latest_taxi_id) # add to available taxi matrix
        self.taxis_available.append(self.latest_taxi_id)
        
        # increase id
        self.latest_taxi_id+=1

    def add_request(self,request_id):
        """
        Create new request.
        
        """
        # here we randonly choose a place for the request
        # origin
        ox,oy = self.city.create_request_coords()
        # destination
        dx,dy = self.city.create_request_coords()
        
        r = Request([ox,oy],[dx,dy],self.latest_request_id,self.time)
        self.latest_request_id+=1
        
        # add to request storage
        self.requests[self.latest_request_id]=r 
        # add to free users
        self.requests_pending.append(self.latest_request_id)
        
    def go_to_base(self, taxi_id ,bcoords):
        """
        This function sends the taxi to the base rom wherever it is.
        """
        # actual coordinates
        acoords = [self.taxis[taxi_id].x,self.taxis[taxi_id].y]
        # path between actual coordinates and destination
        path = self.city.create_path(acoords,bcoords)
        # erase path memory
        self.taxis[taxi_id].next_destination = Queue()
        # put path into taxi path queue
        for p in path:
            self.taxis[taxi_id].next_destination.put(p)
        
    def assign_request(self, request_id):
        """
        This is a sample matching function that assigns the nearest available
        taxi to a user with a random (Gauss) destination.
        """
        # select user
        r = self.requests[request_id]
        
        # search for nearest free taxi
        possible_taxi_ids = self.find_nearest_available_taxis([r.ox,r.oy])
        
        # if there was one
        if len(possible_taxi_ids)>0:
            # select taxi
            taxi_id = choice(possible_taxi_ids)
            t = self.taxis[taxi_id]
            
            # pair the match
            t.actual_request_executing=request_id
            r.taxi_id=taxi_id
            
            # remove taxi from the available ones
            self.city.A[t.x,t.y].remove(taxi_id)
            self.taxis_available.remove(taxi_id)
            t.available=False
            t.to_request=True
            
            # mark taxi as moving to request
            self.taxis_to_request.append(taxi_id)

            # forget the path that has been assigned
            t.next_destination = Queue()
            # create new path: to user, then to destination
            path = self.city.create_path([t.x,t.y],[r.ox,r.oy])+\
                self.city.create_path([r.ox,r.oy],[r.dx,r.dy])[1:]
            for p in path:
                t.next_destination.put(p)
                
            # update taxi state in taxi storage
            self.taxis[taxi_id]=t
            
            # remove it from the free users place    
            self.requests_pending.remove(request_id)
            self.requests_in_progress.append(request_id)
            
            # update request state
            self.requests[request_id]=r
    
        
    def pickup_request(self, request_id):
        """
        Pick up passenger.
        
        Parameters
        ----------
        
        request_id : int
        """
        
        # mark pickup timestamp
        r=self.requests[request_id]
        r.pickup_timestamp=self.time
        t=self.taxis[r.taxi_id]
        
        self.taxis_to_request.remove(r.taxi_id)
        self.taxis_to_destination.append(r.taxi_id)
        
        # change taxi state to with passenger
        t.to_request=False
        t.with_passenger=True
        
        # update request and taxi instances
        self.requests[request_id]=r
        self.taxis[r.taxi_id]=t
        print('\tP '+"request "+str(request_id)+' taxi '+str(t.taxi_id))
    
    def dropoff_request(self,request_id):
        """
        Drop off passenger.
        
        """
        
        # mark dropoff timestamp
        r=self.requests[request_id]
        r.dropoff_timestamp=self.time
        t=self.taxis[r.taxi_id]
        
        # change taxi state to available
        t.with_passenger=False
        t.available=True
        t.actual_request_executing=None
        
        # update taxi lists
        self.city.A[t.x,t.y].append(r.taxi_id)
        self.taxis_to_destination.remove(r.taxi_id)
        self.taxis_available.append(r.taxi_id)
        
        # udate request lists
        self.requests_in_progress.remove(request_id)
        self.requests_fulfilled.append(request_id)
        
        # update request and taxi instances
        self.requests[request_id]=r
        self.taxis[r.taxi_id]=t
        print("\tD request "+str(request_id)+' taxi '+str(t.taxi_id))
    
    def find_nearest_available_taxis(
            self,
            source,
            mode = "nearest",
            radius = None):
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
        while (distance<=self.hard_limit):
            # check available taxis in given nodes
            for x,y in list(frontier):
                visited.append((x,y)) # mark the coordinate as visited
                for t in self.city.A[x,y]: # get all available taxis there
                    possible_plate_numbers.append(t)
            # if we visited everybody, break
            if len(visited)==self.city.n*self.city.m:
                print("\tNo available taxis at this timepoint!")
                break
            # if we got available taxis in nearest mode, break
            if ((mode=="nearest") and (len(possible_plate_numbers)>0)):
                break
            # if we reached our desired depth, break
            if ((mode=="circle") and (distance>radius)):
                break
            # if not, move on to next depth
            else:
                new_frontier = set()
                for f in frontier:
                    new_frontier = \
                        new_frontier.union(self.city.neighbors(f)).difference(set(visited))
                frontier = list(new_frontier)
                distance+=1

        return possible_plate_numbers

    def plot_taxis_w_paths(self):
        for i,taxi in enumerate(self.all_taxi):
            self.canvas_ax.plot(taxi.x,taxi.y,'o',ms=10,c=self.cmap(self.taxi_colors[i]))
            self.canvas_ax.annotate(
                    str(i),
                    xy = (taxi.x,taxi.y),
                    xytext = (taxi.x+0.2,taxi.y+0.2),
                    ha='center',
                    va='center',
                    color=self.cmap(self.taxi_colors[i])
                )
            path=np.array([[taxi.x,taxi.y]]+list(taxi.next_destination.queue))
            if len(path)>1:
                xp,yp=path.T
                self.canvas_ax.plot(
                        xp,
                        yp,
                        '-',
                        c=self.cmap(self.taxi_colors[i])
                        )
                self.canvas_ax.plot(path[-1][0],path[-1][1],'*',ms=5,c=self.cmap(self.taxi_colors[i]))

#    def plot_users(self):
#        for i,user in enumerate(self.all_user):
#            self.canvas_ax.plot(
#                    user.x,
#                    user.y,
#                    'ro',
#                    ms=3)
#            self.canvas_ax.annotate(
#                    str(i),
#                    xy=(user.x,user.y),
#                    xytext=(user.x-0.2,user.y-0.2),
#                    ha='center',
#                    va='center')


    def step_time(self):

#        self.init_canvas()
#        self.plot_taxis_w_paths()
#        self.plot_users()
#        self.canvas.show()
        print("timestamp "+str(self.time))
        test = np.random.rand()
        if test<self.request_rate:
            self.add_request(self.latest_request_id)
        
        # match requests to taxis, if possible
        rp_list = self.requests_pending
        for request_id in rp_list:
            self.assign_request(request_id)
        
        # check for pickups
        tr_list = self.taxis_to_request
        for taxi_id in tr_list:
            t = self.taxis[taxi_id]
            r = self.requests[t.actual_request_executing]
            
            if ((t.x==r.ox) and (t.y==r.oy)):
                self.pickup_request(t.actual_request_executing)
            
        # move every taxi one step towards its destination, or stay in place!
        for i,taxi_id in enumerate(self.taxis.keys()):
            t = self.taxis[taxi_id]
            # if a taxi with a passenger has arrived
            if ((t.next_destination.qsize()==0) and (t.with_passenger)):
                self.dropoff_request(t.actual_request_executing)
                self.go_to_base(taxi_id,self.city.base_coords)
            try:
                # move taxi one step forward
                move = t.next_destination.get_nowait()
                t.x = move[0]
                t.y = move[1]
                # update taxi instance
                self.taxis[taxi_id]=t
            except:
                pass

        # step time
        self.time+=1