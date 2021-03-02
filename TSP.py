import os
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randrange

kroA100_filename = 'kroA100.tsp'
kroB100_filename = 'kroB100.tsp'

test_points = [(1,1), (1.5,5), (3.5,2), (5,1.5),
               (5.5,5), (7.5,2), (8,3.5), (10.5, 1),
               (20, 10.5), (15, 13), (8,10), (10, 12),
               (7, 8), (10, 7), (11, 5), (15,7)]

class TSP_solver:
    def __init__(self, filename):
        #self.coords = self.get_graph_from_tsp(filename)
        self.coords = test_points
        print(self.coords)
        self.dist_matrix = self.get_distance_matrix(self.coords)
        print(self.dist_matrix)

    # load point coordinates from tsp file
    def get_graph_from_tsp(self, filename):
        fullpath = os.path.abspath(os.path.join('data', filename))
        file = open(fullpath, "r+")
        lines = file.readlines()
        file.close()
        lines = lines[6:-1]

        coords = []
        for point in lines:
            p = point.split()
            coords.append(np.array([int(p[1]), int(p[2])]))
        return coords

    # calculate distance matrix from coordinates
    def get_distance_matrix(self, coords):
        matrix = []
        for coord in coords:
            v = []
            for next_coord in coords:
                dist = 0
                if not(np.array_equal(coord, next_coord)):
                    dist = int(round(math.sqrt(((coord[0]-next_coord[0])**2)+((coord[1]-next_coord[1])**2))))
                    #dist = int(round(np.linalg.norm(coord - next_coord)))
                v.append(dist)
            matrix.append(v)
        return matrix

    # draw the graph
    def draw_graph(self, title, edge_matrix):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.scatter([p[0] for p in self.coords], [p[1] for p in self.coords])

        def draw_edge(a, b):
            ax.annotate("",
                        xy=a, xycoords='data',
                        xytext=b, textcoords='data',
                        arrowprops=dict(arrowstyle="-",
                                        connectionstyle="arc3"))

        # draw edges
        for x in range(len(edge_matrix)):
            start_pos = self.coords[x]
            end_pos = self.coords[edge_matrix[x]]
            draw_edge(start_pos, end_pos)

        # show the graph
        distance = self.get_cycle_length(edge_matrix)
        N = len(self.coords)
        textstr = "N nodes: %d\nTotal length: %d" % (N, distance)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()

    def get_cycle_length(self, edge_matrix):
        cycle_len = 0
        for x in edge_matrix:
            edge_cost = self.dist_matrix[x][edge_matrix[x]]
            cycle_len = cycle_len+edge_cost
        return cycle_len

    def get_starting_points(self):
        start1 = randrange(len(self.coords))
        distances = self.dist_matrix[start1]
        start2 = distances.index(max(distances))
        return start1, start2

    def greedy_nearest_neighbor(self):
        def find_nearest_neighbor(point, visited):
            distances = self.dist_matrix[point]
            #print("DISTANCES OF ", point, distances)

            min_d = min(i for i in distances if distances.index(i) not in visited)

            #print("NEAREST NEIGHBOR OF ", point, distances.index(min_d), ", distance:", min_d)
            return distances.index(min_d)

        start1, start2 = self.get_starting_points()
        edge_matrix=[]
        for i in range(len(self.coords)):
            edge_matrix.append(0)
        print("STARTING POINTS: ", start1, start2)
        visited = [start1, start2]


        current1 = start1
        current2 = start2
        while len(visited) < len(self.coords):
            nn = find_nearest_neighbor(current1, visited)
            edge_matrix[current1] = nn
            current1 = nn
            visited.append(current1)
            nn = find_nearest_neighbor(current2, visited)
            edge_matrix[current2] = nn
            current2 = nn
            visited.append(current2)
        edge_matrix[current1] = start1
        edge_matrix[current2] = start2

        #print("VISITED: ", visited)
        #print("EDGE MATRIX: ", edge_matrix)
        self.draw_graph('Greedy nearest neighbor', edge_matrix)
        return edge_matrix

    def greedy_cycle(self):
        start1, start2 = self.get_starting_points()
        edge_matrix=[]
        for i in range(len(self.coords)):
            edge_matrix.append(0)

        self.draw_graph('Greedy cycle', edge_matrix)
        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        return edge_matrix

    def regret(self):
        start1, start2 = self.get_starting_points()
        edge_matrix=[]
        for i in range(len(self.coords)):
            edge_matrix.append(0)

        self.draw_graph('2-regret', edge_matrix)
        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        return edge_matrix


solver = TSP_solver(kroA100_filename)
