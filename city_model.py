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
            self.A = np.empty((self.n,self.m),dtype=list) # array that stores plate_no of available taxis at the specific position
            for i in range(self.n):
                for j in range(self.m):
                    self.A[i,j]=list()
                    
    def measure_distance(self,source,destination):
        """
        MEasure distance on the grid between two points.
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
        Edges are the end of the world.
        
        """
        
        ns = set()
        for dx,dy in [(1,0),(0,1),(-1,0),(0,-1)]:
            new_x = coordinates[0]+dx
            new_y = coordinates[1]+dy
            
            if ((0<=new_x) and (self.n>new_x) and (0<=new_y) and (self.m>new_y)):
                ns.add((new_x,new_y))
        return ns
                
class Taxi():
    """
    Represents a taxi in the simulation.
    
    """
    def __init__(self,coordinates = None,plate_no = None):
        if coordinates == None:
            print("You have to put your taxi somewhere in the city!")
        elif plate_no == None:
            print("Not a licenced taxi.")
        else:
            self.x=coordinates[0]
            self.y=coordinates[1]
            self.plate_no=plate_no
            self.actual_passenger=None
            self.passenger_to_pick_up=None
            
            # metrics
            self.waiting_time=0
            self.useful_travel_time=0
            self.empty_travel_time=0
            
            # storing steps to take
            self.next_destination=Queue() # path to travel
    
class User():
    """
    Represents a user in the simulation.
    """
    
    def __init__(self,coordinates = None,user_id = None):
        if coordinates == None:
            print("You have to put your user somewhere in the city!")
        elif user_id == None:
            print("A user cannot live without a name.")
        else:
            self.x=coordinates[0]
            self.y=coordinates[1]
            
            self.user_id=user_id
            self.travelling_with=None # this is going to store the taxi_id


class Simulation():
    """
    Class for containing the elements of the simulation.
    """
    
    def __init__(self,**config):
        self.time = 0
        
        self.num_users = config["num_users"]
        self.num_taxis = config["num_taxis"]
        
        self.all_taxi = []
        self.all_user = []
        self.free_users = []
        
        self.taxis_towards_users = []
        self.taxis_with_users = []
        
        self.geometry = City(**config)
                
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
        self.canvas_ax.set_xlim(-0.5,self.geometry.n-0.5)
        self.canvas_ax.set_ylim(-0.5,self.geometry.m-0.5)
        
        self.canvas_ax.xaxis.set_ticks(list(range(self.geometry.n)))
        self.canvas_ax.yaxis.set_ticks(list(range(self.geometry.m)))
        
        self.canvas_ax.grid()
                
    def add_taxi(self,plate_no):
        x = np.random.randint(0,self.geometry.n)
        y = np.random.randint(0,self.geometry.m)
        
        tx = Taxi([x,y],plate_no)
        
        self.all_taxi.append(tx) # add to taxi storage
        self.geometry.A[x,y].append(plate_no) # add to available taxi matrix
        
    def add_user(self,user_id):
        x = np.random.randint(0,self.geometry.n)
        y = np.random.randint(0,self.geometry.m)
        
        u = User([x,y],user_id)
        self.all_user.append(u) # add to user storage
        self.free_users.append(user_id) # add to free users
        
    def find_nearest_available_taxi(self,source):
        frontier = [source]
        visited = []
        possible_plate_numbers = []
        
        while True:
            # check available taxis in given nodes
            for x,y in list(frontier):
                visited.append((x,y)) # mark the coordinate as visited
                for t in self.geometry.A[x,y]: # get all available taxis there
                    possible_plate_numbers.append(t)
            # if we visited everybody, break
            if len(visited)==self.geometry.n*self.geometry.m:
                print("No available taxis at this timepoint!")
                break
            # if we got what we wanted (e.g. available taxis), break
            if len(possible_plate_numbers)>0:
                break
            # if not, move on to next depth
            else:
                new_frontier = set()
                for f in frontier:
                    new_frontier = \
                        new_frontier.union(self.geometry.neighbors(f)).difference(set(visited))
                frontier = list(new_frontier)     
        if len(possible_plate_numbers)>0:
            # assign a taxi randomly from the nearest ones
            return choice(possible_plate_numbers)
        else:
            return None
        
    def make_request(self,user_id,user_destination):
        # select user
        user = self.all_user[user_id]
        # get its position
        source = [user.x,user.y]
        # search for nearest free taxi
        taxi_plate_no = self.find_nearest_available_taxi(source)
        # if there was one
        if taxi_plate_no!=None:
            # select taxi
            taxi = self.all_taxi[taxi_plate_no]
            
            # append taxi to occupied but empty ones
            self.taxis_towards_users.append(taxi_plate_no)
            taxi.passenger_to_pick_up=user_id
            # remove taxi from the available ones
            self.geometry.A[taxi.x,taxi.y].remove(taxi_plate_no)
            
            path = self.geometry.create_path([taxi.x,taxi.y],source)+\
                self.geometry.create_path(source,user_destination)[1:]
            for p in path: 
                taxi.next_destination.put(p)                
            # update taxi state in taxi storage
            self.all_taxi[taxi_plate_no]=taxi
            # remove it from the free users place
            self.free_users.remove(user_id)
    
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
        
        self.init_canvas()
        self.plot_taxis_w_paths()
        self.plot_users()
        self.canvas.show()
        
        # make new requests
        if len(self.free_users)>0:
#            print("Free users are "+",".join(list(map(str,self.free_users))))
            p = choice(self.free_users)
            
            # assign available taxis to requests
            x = np.random.randint(0,high=self.geometry.n)
            y = np.random.randint(0,high=self.geometry.m)
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
#                  t.plate_no,
#                  "actual passenger",
#                  t.actual_passenger,
#                  "moving towards",
#                  t.passenger_to_pick_up)
            # if the taxi has arrived and there was somebody sitting in it
            if ((self.all_taxi[i].next_destination.qsize()==0)\
                and (self.all_taxi[i].actual_passenger!=None)):
                print("Dropping off my passenger! Taxi no "+\
                      str(self.all_taxi[i].plate_no)+' passenger '+\
                      str(self.all_taxi[i].actual_passenger)+'.')
                self.free_users.append(self.all_taxi[i].actual_passenger)
                self.all_user[self.all_taxi[i].actual_passenger].travelling_with=None
                self.all_taxi[i].actual_passenger=None
                # write taxi again on availability map
                self.geometry.A[self.all_taxi[i].x,self.all_taxi[i].y]\
                .append(self.all_taxi[i].plate_no)
                self.taxis_with_users.remove(self.all_taxi[i].plate_no)
            
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
        