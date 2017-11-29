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
                    
            self.base_coords = [np.floor(self.n/2),np.floor(self.m/2)]
            self.base_sigma = config["base_sigma"]

    def measure_distance(self,source,destination):
        """
        Measure distance on the grid between two points.
        |xs-xd|+|ys-yd|
        """

        return np.dot(np.abs(np.array(destination)-np.array(source)),[1,1])

    def create_path(self,source,destination):
        """
        Choose a random shortest path between source and dest.
        """

        dx, dy = np.array(destination) - np.array(source)
        sequence = ['x']*int(np.abs(dx))+['y']*int(np.abs(dy))
        shuffle(sequence)
        path=[source] # appending source
        for item in sequence:
            if item=="x":
                path.append([np.sign(dx)+path[-1][0],0+path[-1][1]])
            else:
                path.append([0+path[-1][0],np.sign(dy)+path[-1][1]])
        return path

    def plot_path(self,source,destination,path):
        """
        Plot selected path in a grid.
        """

        path=np.array(path)
        path_to_plot = np.cumsum(path,axis=0)+np.array(source)*np.ones(np.shape(path))
        x,y=path_to_plot.T
        plt.plot(x,y,'bo-')


    def neighbors(self,coordinates):
        """
        Return the neighbors of a coordinate.
        On the edges of the simulation grid, there are no neighbors.
        (E.g. there are only 2 neighbors in the corners.)

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
        """
        
        done=False
        while(done):
            mean = self.base_coords
            cov = np.array([[self.base_sigma**2,0],[0,self.base_sigma**2]])
            
            x,y=np.random.multivariate_normal(mean,cov)
            
            x = np.floor(x)
            y = np.floor(y)
            
            if ((x>=0) and (x<self.n) and (y>=0) and (y<self.m)):
                done=True
        return x,y

class Taxi():
    """
    Represents a taxi in the simulation.

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
    """

    def __init__(self,**config):
        
        # initializing empty grid
        self.time = 0

        self.num_taxis = config["num_taxis"]
        self.request_rate = config["request_rate"]

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

        # initializing simulation with users
        for u in range(self.num_users):
            self.add_user(u)
        for t in range(self.num_taxis):
            self.add_taxi(t)

        # plotting variables
        self.canvas = plt.figure()
        self.canvas_ax = self.canvas.add_subplot(1,1,1)
        self.cmap = plt.get_cmap('viridis')
        self.taxi_colors = np.random.rand(self.num_taxis)
        self.init_canvas()

    def init_canvas(self):
        self.canvas_ax.clear()
        self.canvas_ax.set_xlim(-0.5,self.city.n-0.5)
        self.canvas_ax.set_ylim(-0.5,self.city.m-0.5)

        self.canvas_ax.xaxis.set_ticks(list(range(self.city.n)))
        self.canvas_ax.yaxis.set_ticks(list(range(self.city.m)))

        self.canvas_ax.grid()

    def add_taxi(self,taxi_id):

        # create a taxi at the base
        tx = Taxi(self.city.base_coords,self.latest_taxi_id)
        
        # add to taxi storage
        self.taxis[self.lates_taxi_id]= tx
        # add to available taxi storage
        self.city.A[self.city.base_coords[0],self.city.base_coords[1]].append(self.latest_taxi_id) # add to available taxi matrix
        self.taxis_available.append(self.latest_taxi_id)
        
        # increase id
        self.latest_taxi_id+=1

    def add_request(self,request_id):
        # here we randonly choose a place for the request
        x,y = self.city.create_request_coords()

        r = Request([x,y],self.latest_request_id)
        self.latest_request_id+=1
        
        # add to request storage
        self.requests[self.latest_request_id]=r 
        # add to free users
        self.requests_pending.append(self.latest_request_id)
        
    def go_to_base(self, taxi_id ,bcoords):
    
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
        possible_taxi_ids = self.find_nearest_available_taxi([r.ox,r.oy])
        
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
                self.city.create_path(r.ox,r.oy,r.dx,r.dy)[1:]
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
    
    def dropoff_request(self,request_id):
        # mark dropoff timestamp
        r=self.requests[request_id]
        r.dropoff_timestamp=self.time
        t=self.taxis[r.taxi_id]
        
        # change taxi state to available
        t.with_passenger=False
        t.available=True
        
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
    
    def find_nearest_available_taxis(self, source, mode = "nearest", radius = None):
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
        while True:
            # check available taxis in given nodes
            for x,y in list(frontier):
                visited.append((x,y)) # mark the coordinate as visited
                for t in self.city.A[x,y]: # get all available taxis there
                    possible_plate_numbers.append(t)
            # if we visited everybody, break
            if len(visited)==self.city.n*self.city.m:
                print("No available taxis at this timepoint!")
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

    def plot_users(self):
        for i,user in enumerate(self.all_user):
            self.canvas_ax.plot(
                    user.x,
                    user.y,
                    'ro',
                    ms=3)
            self.canvas_ax.annotate(
                    str(i),
                    xy=(user.x,user.y),
                    xytext=(user.x-0.2,user.y-0.2),
                    ha='center',
                    va='center')


    def step_time(self):

#        self.init_canvas()
#        self.plot_taxis_w_paths()
#        self.plot_users()
#        self.canvas.show()

        # make new requests with request rate
        if len(self.free_users)>0:
#            print("Free users are "+",".join(list(map(str,self.free_users))))
            p = choice(self.free_users)

            # assign available taxis to requests
            x = np.random.randint(0,high=self.city.n)
            y = np.random.randint(0,high=self.city.m)
            self.make_request(p,[x,y])

        # passenger pickup
        pickups=[]
        for t in self.taxis_towards_users:
            user_pick = self.all_taxi[t].passenger_to_pick_up

            if ((self.all_taxi[t].x==self.all_user[user_pick].x) \
                and (self.all_taxi[t].y==self.all_user[user_pick].y)):
                print('I\'ve picked up user '+\
                      str(user_pick)+\
                      ' with taxi '+\
                      str(t)+'!')
                pickups.append(t)
                self.all_taxi[t].actual_passenger=user_pick
                self.all_taxi[t].passenger_to_pick_up=None
                self.all_user[user_pick].travelling_with=t
        for p in pickups:
            self.taxis_towards_users.remove(p)
            self.taxis_with_users.append(p)

        # move every taxi one step towards its destination, or stay in place!
        for i,t in enumerate(self.all_taxi):

#            print("Taxi ",
#                  t.taxi_id,
#                  "actual passenger",
#                  t.actual_passenger,
#                  "moving towards",
#                  t.passenger_to_pick_up)
            # if the taxi has arrived and there was somebody sitting in it
            if ((self.all_taxi[i].next_destination.qsize()==0)\
                and (self.all_taxi[i].actual_passenger!=None)):
                print("Dropping off my passenger! Taxi no "+\
                      str(self.all_taxi[i].taxi_id)+' passenger '+\
                      str(self.all_taxi[i].actual_passenger)+'.')
                self.free_users.append(self.all_taxi[i].actual_passenger)
                self.all_user[self.all_taxi[i].actual_passenger].travelling_with=None
                self.all_taxi[i].actual_passenger=None
                # write taxi again on availability map
                self.city.A[self.all_taxi[i].x,self.all_taxi[i].y]\
                .append(self.all_taxi[i].taxi_id)
                self.taxis_with_users.remove(self.all_taxi[i].taxi_id)

            try:
                # move taxi one step forward
                move = self.all_taxi[i].next_destination.get_nowait()
                self.all_taxi[i].x = move[0]
                self.all_taxi[i].y = move[1]
                # move user along with taxi
                if self.all_taxi[i].actual_passenger!=None:
                    self.all_user[self.all_taxi[i].actual_passenger].x=t.x
                    self.all_user[self.all_taxi[i].actual_passenger].y=t.y

            except:
                pass


        # step time
        self.time+=1
        print(self.time)
