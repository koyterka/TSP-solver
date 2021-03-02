import os
import numpy as np
import matplotlib.pyplot as plt

kroA100_filename = 'kroA100.tsp'
kroB100_filename = 'kroB100.tsp'


class TSP_solver:
    def __init__(self, filename):
        self.coords = self.get_graph_from_tsp(filename)
        self.dist_matrix = self.get_distance_matrix(self.coords)

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
                dist = int(np.linalg.norm(coord - next_coord))
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
        distance = 0
        N = len(self.coords)
        textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()

    def greedy_nearest_neighbor(self):

        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        edge_matrix = []
        self.draw_graph('Greedy nearest neighbor', edge_matrix)
        return edge_matrix

    def greedy_cycle(self):

        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        edge_matrix = []
        self.draw_graph('Greedy cycle', edge_matrix)
        return edge_matrix

    def regret(self):

        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        edge_matrix = []
        self.draw_graph('2-regret', edge_matrix)
        return edge_matrix


solver = TSP_solver(kroA100_filename)
solver.greedy_cycle()
