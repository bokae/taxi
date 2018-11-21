
import numpy as np
from random import shuffle, gauss, random
import json

# special data types
from collections import deque
from queue import Queue

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

        request_origin_distributions : list of dicts
            stores the distribution of request origins on the grid
            describes the 2D Gaussians from which the total distribution is created
            each Gaussian is given by its location, standard deviation and strength
                {"location":[int,int], "sigma":float, "strength":float}
            



        Attributes
        ----------
        A : np.array of lists
            placeholder to store list of available taxi_ids at their actual positions

        N : np.array of sets
            stores neighborhoods for faster access

        n : int
            width of grid

        m : int
            height of grid

        length : int
            length of coordstacks that help in random origin and destination generation
        """

        # grid dimensions
        self.n = config["n"]  # number of pixels in x direction
        self.m = config["m"]  # number of pixels in y direction

        if "base_coords" in config:
            self.base_coords = config["base_coords"]
        else:
            self.base_coords = [int(self.n/2), int(self.m/2)]

        # list that stores taxi_id of available taxis at the
        # specific position on the grid
        # we initialize this array with empty sets
        self.A = [set() for _ in range(self.n*self.m)]

        # storing neighbors
        self.N = {c: self.neighbors(c) for c in range(self.n*self.m)}

        # generating stacks for request coordinate choice

        self.request_p = deque([])

        if "base_sigma" in config:
            self.request_origin_distributions = [
                {"location": self.base_coords, "sigma": config["base_sigma"], "strength": 1}]

        if 'request_origin_distributions' in config:
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

        self.taxi_home_coordstack = deque([])

        if "hard_limit" in config:
            self.hard_limit = config["hard_limit"]
        else:
            self.hard_limit = self.n+self.m

        if 'length' not in config:
            self.length = int(2e5)

        # pre-storing coordinates
        # dicts for converting between real positions and their labels
        self.coordinate_dict_ij_to_c = {}
        for i in range(self.m):
            for j in range(self.n):
                if i in self.coordinate_dict_ij_to_c:
                    self.coordinate_dict_ij_to_c[i][j]=self.ij_to_c(i,j)
                else:
                    self.coordinate_dict_ij_to_c[i] = {j : self.ij_to_c(i,j)}

        self.coordinate_dict_c_to_ij = {}
        for c in range(self.n*self.m):
            self.coordinate_dict_c_to_ij[c] = self.c_to_ij(c)

        # pre-storing BFS-trees until the depth self.hard_limit
        self.bfs_trees = {}
        for c in range(self.n*self.m):
            self.bfs_trees[c] = self.create_BFS_tree(c)

    def create_one_request_coord(self):
        # here we randomly choose an origin and a destination for the request
        # the random coordinates are pre-stored in several deques for faster access
        # if there are no more pregenerated coordinates in the deques, we generate some more
        # probability stack
        try:
            p = self.request_p.pop()
        except IndexError:
            self.request_p.extend(np.random.random(self.length))
            p = self.request_p.pop()

        # binning the generated random numbers
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

    def neighbors(self, c):
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

        coordinates = self.c_to_ij(c)

        ns = [(coordinates[0] + dx, coordinates[1] + dy) for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        ns = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]), ns)

        return [self.ij_to_c(x, y) for x, y in ns]

    def ij_to_c(self, i, j):
        # grid coordinates to continuous coordinates
        return self.n*i+j

    def c_to_ij(self, c):
        # continuous coordinates to grid coordinates
        return int(c/self.n), c % self.n

    #@profile
    def generate_coords(self, **gauss_spec):

        temp = map(
            lambda t: (int(round(t[0]*gauss_spec["sigma"], 0)+gauss_spec["location"][0]),
                       int(round(t[1]*gauss_spec["sigma"], 0))+gauss_spec["location"][1]),
            np.random.normal(size=(self.length, 2))
        )
        temp = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]), temp)

        return temp

    def create_taxi_home_coords(self):
        try:
            hx,hy = self.taxi_home_coordstack.pop()
        except IndexError:
            temp = list(zip(np.random.randint(0, self.n, 1000), np.random.randint(0, self.m, 1000)))
            self.taxi_home_coordstack.extend(temp)
            hx,hy = self.taxi_home_coordstack.pop()
        return hx,hy

    #@profile
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
        if self.log:
            print("Finding nearest available taxi.")
        if mode == "nearest":
            radius = self.hard_limit

        # select BFS-tree
        tree = self.bfs_trees[source]
        # print('This is the BFS-tree.', tree)

        # current depth storage
        depth = 0
        # list of available taxis

        p = set()

        while depth < radius:
            # take the next nodes
            # print('Taxis available at depth ' + str(depth)+' at nodes '+json.dumps(tree[depth]))
            ta = set.union(*[self.A[node] for node in tree[depth]])
            # print(ta)
            # print(self.A)
            p = p.union(ta)

            if mode == "nearest" and len(ta)>0:
                # print("I've found something!")
                break

            depth += 1

        return list(p)


    def create_BFS_tree(
            self,
            source,
            max_depth = None
    ):

        if max_depth == None:
            max_depth = self.hard_limit+1

        # BFS init
        # queue for BFS visit
        q = deque()
        q.append(source)
        # visited nodes with distance from the source node
        visited = {source: 0}
        # current depth storage
        depth = 0

        # while we still have nodes to visit
        while len(q) != 0 and depth < max_depth:
            # take the next node
            v = q.popleft()

            # visit the neighbors of v
            for n in self.N[v]:
                if n not in visited:
                    q.append(n)
                    # the depth of the neighbor is one more than that of its parent in the BFS tree
                    depth = visited[v] + 1
                    # if we surpass the search radius, quit BFS
                    if depth > max_depth:
                        break
                    # store node in the visited ones
                    visited[n] = depth

        bfs_tree = {}
        # reverse dict
        for k,v in visited.items():
            if v in bfs_tree:
                bfs_tree[v].append(k)
            else:
                bfs_tree[v] = [k]

        return bfs_tree