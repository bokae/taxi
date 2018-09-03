
import numpy as np
from random import shuffle, gauss, random

# special data types
from collections import deque


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
        """

        # grid dimensions
        self.n = config["n"]  # number of pixels in x direction
        self.m = config["m"]  # number of pixels in y direction

        self.base_coords = config["base_coords"]

        # array that stores taxi_id of available taxis at the
        # specific position on the grid
        # we initialize this array with empy lists
        self.A = np.empty((self.n, self.m), dtype=set)
        for i in range(self.n):
            for j in range(self.m):
                self.A[i, j] = set()

        # storing neighbors
        self.N = np.empty((self.n, self.m), dtype=set)
        for i in range(self.n):
            for j in range(self.m):
                self.N[i, j] = self.neighbors((i, j))

        # generating stacks for request coordinate choice

        # probabilities
        self.length= int(2e5)

        self.request_p = deque([])


        # origins
        self.request_origin_distributions = config['request_origin_distributions']
        self.request_origin_coordstacks = \
            [deque([]) for i in range(len(self.request_origin_distributions))]
        self.request_origin_strengths = \
            [distr["strength"] for distr in self.request_origin_distributions]
        self.request_origin_probabilities = \
            np.cumsum(np.array(self.request_origin_strengths)/sum(self.request_origin_strengths))

        # destinations, if different from origins
        if 'request_destination_distributions' in config:
            self.request_destination_distributions = \
                config['request_destination_distributions']
            self.request_destination_coordstacks = \
                [deque([]) for i in range(len(self.request_destination_distributions))]
            self.request_destination_strengths = \
                [distr["strength"] for distr in self.request_destination_distributions]
            self.request_destination_probabilities = \
                np.cumsum(np.array(self.request_destination_strengths)/sum(self.request_destination_strengths))
        else:
            self.request_destination_distributions = \
                self.request_origin_distributions
            self.request_destination_coordstacks = \
                self.request_origin_coordstacks
            self.request_destination_probabilities = \
                np.cumsum(self.request_origin_probabilities)

    def create_one_request_coord(self):
        # here we randomly choose an origin and a destination for the request
        # the random coordinates are pre-stored in several deques for faster access
        # if there are no more pregenerated coordinates in the deques, we generate some more

        # origin
        try:
            p = self.request_p.pop()
        except IndexError:
            self.request_p.extend(np.random.random(self.length))
            p = self.request_p.pop()

        ind = np.digitize(p, self.request_origin_probabilities)

        try:
            ox, oy = self.request_origin_coordstacks[ind].pop()
        except IndexError:
            self.request_origin_coordstacks[ind].extend(
                self.generate_coords(**self.request_origin_distributions[ind])
            )
            ox, oy = self.request_origin_coordstacks[ind].pop()

        # destination
        try:
            p = self.request_p.pop()
        except IndexError:
            self.request_p.extend(np.random.random(2e5))
            p = self.request_p.pop()

        ind = np.digitize(p, self.request_destination_probabilities)

        try:
            dx, dy = self.request_destination_coordstacks[ind].pop()
        except IndexError:
            self.request_destination_coordstacks[ind].extend(
                self.generate_coords(**self.request_destination_distributions[ind])
            )
            dx, dy = self.request_destination_coordstacks[ind].pop()

        return ox, oy, dx, dy

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
        d = dict(zip(['x', 'y'], np.array(destination) - np.array(source)))

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
                np.sign(d[item]) * int(item == "x") + path[-1][0],
                np.sign(d[item]) * int(item == "y") + path[-1][1]
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
        ns = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]), ns)

        return set(ns)

    def generate_coords(self, **gauss_spec):

        temp = map(
            lambda t: (int(round(t[0], 0)), int(round(t[1], 0))),
            np.random.multivariate_normal(gauss_spec["location"],gauss_spec["sigma"]*np.eye(2, 2), self.length)
        )
        temp = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]), temp)

        return temp
