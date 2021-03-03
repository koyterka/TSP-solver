import os
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randrange

kroA100_filename = 'kroA100.tsp'
kroB100_filename = 'kroB100.tsp'

test_points = [(1, 1), (1.5, 5), (3.5, 2), (5, 1.5),
               (5.5, 5), (7.5, 2), (8, 3.5), (10.5, 1),
               (20, 10.5), (15, 13), (8, 10), (10, 12),
               (7, 8), (10, 7), (11, 5), (15, 7)]


# test_points = [(1,1), (1.5,5), (3.5,2), (5,1.5),
#                (5.5,5)]

class TSP_solver:
    def __init__(self, filename):
        self.coords = self.get_graph_from_tsp(filename)
        #self.coords = test_points
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
                dist = 0
                if not (np.array_equal(coord, next_coord)):
                    dist = int(round(math.sqrt(((coord[0] - next_coord[0]) ** 2) + ((coord[1] - next_coord[1]) ** 2))))
                v.append(dist)
            matrix.append(v)
        return matrix

    # draw the graph
    def draw_graph(self, title, path1, path2):
        fig, ax = plt.subplots()
        ax.set_title(title)
        path1_coords = [self.coords[x] for x in path1]
        path2_coords = [self.coords[x] for x in path2]
        ax.scatter([p[0] for p in path1_coords], [p[1] for p in path1_coords], c='#ff0e8e')
        ax.scatter([p[0] for p in path2_coords], [p[1] for p in path2_coords], c='#0eff7f')

        def draw_edge(a, b):
            ax.annotate("",
                        xy=a, xycoords='data',
                        xytext=b, textcoords='data',
                        arrowprops=dict(arrowstyle="-",
                                        connectionstyle="arc3"))

        # draw path1
        for x in range(len(path1) - 1):
            start_pos = self.coords[path1[x]]
            end_pos = self.coords[path1[x + 1]]
            draw_edge(start_pos, end_pos)
        draw_edge(self.coords[path1[-1]], self.coords[path1[0]])

        # draw path2
        for x in range(len(path2) - 1):
            start_pos = self.coords[path2[x]]
            end_pos = self.coords[path2[x + 1]]
            draw_edge(start_pos, end_pos)
        draw_edge(self.coords[path2[-1]], self.coords[path2[0]])

        # show the graph
        distance = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        N = len(self.coords)
        textstr = "N nodes: %d\nTotal length: %d" % (N, distance)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()

    def get_cycle_length(self, cycle):
        cycle_len = 0
        for x in range(len(cycle) - 1):
            edge_cost = self.dist_matrix[cycle[x]][cycle[x + 1]]
            cycle_len = cycle_len + edge_cost
        return cycle_len

    def get_starting_points(self):
        start1 = randrange(len(self.coords))
        # if we choose the furthest
        distances = self.dist_matrix[start1]
        start2 = distances.index(max(distances))
        # if we choose random
        # start2 = randrange(len(self.coords))
        # while start1 == start2:
        #     start2 = randrange(len(self.coords))
        return start1, start2

    def group_vertices(self, start1, start2):
        # groups vertices into two clusters based on their distance
        # to the starting points
        dist_set = []
        # for each point, get distances from start1 and start2
        # and sort points based on how much closer they are to start1 than to start2
        for point in range(len(self.dist_matrix)):
            dist1 = self.dist_matrix[point][start1]
            dist2 = self.dist_matrix[point][start2]
            if dist1 != 0 and dist2 != 0:
                dist_set.append((point, dist1 / dist2))

        dist_set.sort(key=lambda x: x[1])
        # print("SET OF DISTANCES: ", dist_set)

        mid = int(round(len(dist_set) / 2))
        # put points closer to start1 to cluster1,
        # put points closer to start2 to cluster2
        cluster1 = [start1] + [p[0] for p in dist_set[:mid]]
        cluster2 = [start2] + [p[0] for p in dist_set[mid:]]
        return cluster1, cluster2

    def greedy_nearest_neighbor(self):
        # for point, find the nearest neighbor in cluster that hasn't been visited yet
        def find_nearest_neighbor(point, cluster, visited):
            distances = self.dist_matrix[point]
            min_d = max(distances)
            nn = None
            for x in cluster:
                if x not in visited and distances[x] <= min_d:
                    min_d = distances[x]
                    nn = x
            # min_d = min(i for i in distances if distances.index(i) not in visited)
            return nn

        def find_cycle(cluster, start):
            # in cluster, use nn to find a cycle that starts with start
            visited = [start]
            current = start
            while len(visited) < len(cluster):
                nn = find_nearest_neighbor(current, cluster, visited)
                visited.append(nn)
                current = nn
            visited.append(start)
            return visited

        def get_shortest_cycle(cluster):
            # find the shortest possible cycle in cluster
            paths = []
            while len(paths) < len(cluster):
                path = find_cycle(cluster, cluster[len(paths)])
                paths.append(path)
            paths_cost = []
            for p in range(len(paths)):
                paths_cost.append((p, self.get_cycle_length(paths[p])))
            paths_cost.sort(key=lambda x: x[1])
            shortest_path = paths[paths_cost[0][0]]
            # transform the path so it starts properly
            start = cluster[0]
            if shortest_path[0] != start:
                start_id = shortest_path.index(start)
                shortest_path = shortest_path[start_id:-1] + \
                                shortest_path[:start_id] + [start]
            return shortest_path

        # set starting points
        start1, start2 = self.get_starting_points()
        print("Starting points: ", start1, start2)

        cluster1, cluster2 = self.group_vertices(start1, start2)

        # find the shortest path in cluster1 and cluster2
        path1 = get_shortest_cycle(cluster1)
        path2 = get_shortest_cycle(cluster2)

        # get the path length
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        # draw the paths
        self.draw_graph('Greedy nearest neighbor', path1, path2)
        return path_len

    def greedy_cycle(self):
        start1, start2 = self.get_starting_points()
        edge_matrix = []
        for i in range(len(self.coords)):
            edge_matrix.append(0)

        self.draw_graph('Greedy cycle', edge_matrix)
        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        return edge_matrix

    def regret(self):
        start1, start2 = self.get_starting_points()
        edge_matrix = []
        for i in range(len(self.coords)):
            edge_matrix.append(0)

        self.draw_graph('2-regret', edge_matrix)
        # returns edge matrix [a, b, ..., n]
        # meaning point 0 connects to point a, point 1 to point b...
        return edge_matrix

    def algo_test(self, filename):
        f = open(filename, "w")
        for x in range(100):
            p_len = self.greedy_nearest_neighbor()
            print("test", x, "len:", p_len)
            f.write(str(p_len)+'\n')
        f.close()


solver = TSP_solver(kroA100_filename)
solver.greedy_nearest_neighbor()
