import os
import numpy as np
import math
from TSP_visualiser import TSP_visualiser
from random import randrange

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
                dist = 0
                if not (np.array_equal(coord, next_coord)):
                    dist = int(round(math.sqrt(((coord[0] - next_coord[0]) ** 2) + ((coord[1] - next_coord[1]) ** 2))))
                v.append(dist)
            matrix.append(v)
        return matrix

    # calculate the length of cycle
    def get_cycle_length(self, cycle):
        cycle_len = 0
        for i, point in enumerate(cycle[:-1]):
            edge_len = self.dist_matrix[point][cycle[i + 1]]
            cycle_len = cycle_len + edge_len
        return cycle_len

    # get a random starting point (start1), get the furthest point from starting point (start2)
    def get_starting_points(self, s=None):
        start1 = randrange(len(self.coords))
        if s:
            start1 = s
        distances = self.dist_matrix[start1]
        start2 = distances.index(max(distances))
        # if we choose random
        # start2 = randrange(len(self.coords))
        # while start1 == start2:
        #     start2 = randrange(len(self.coords))
        return start1, start2

    # groups vertices into two clusters based on their distance
    # to the starting points
    def group_vertices(self, start1, start2):
        dist_set = []
        # for each point, get distances from start1 and start2
        # and sort points based on how much closer they are to start1 than to start2
        for point in range(len(self.dist_matrix)):
            dist1 = self.dist_matrix[point][start1]
            dist2 = self.dist_matrix[point][start2]
            if dist1 != 0 and dist2 != 0:
                dist_set.append((point, dist1 / dist2))
        dist_set.sort(key=lambda x: x[1])
        # put points closer to start1 to cluster1,
        # put points closer to start2 to cluster2
        mid = int(round(len(dist_set) / 2))
        cluster1 = [start1] + [p[0] for p in dist_set[:mid]]
        cluster2 = [start2] + [p[0] for p in dist_set[mid:]]
        return cluster1, cluster2

    # for point, find the nearest neighbour in cluster that hasn't been visited yet
    def find_nearest_neighbour(self, point, cluster, visited):
        distances = self.dist_matrix[point]
        min_d = max(distances)
        nn = distances.index(min_d)
        for x in cluster:
            if x not in visited and distances[x] <= min_d:
                min_d = distances[x]
                nn = x
        return nn, min_d

    # for point, find the furthest neighbor in cluster that hasn't been visited yet
    def find_furthest_neighbour(self, point, cluster, visited):
        distances = self.dist_matrix[point]
        max_d = min(distances)
        nn = distances.index(max_d)
        for x in cluster:
            if x not in visited and distances[x] > max_d:
                max_d = distances[x]
                nn = x
        return nn, max_d

    # calculate the cost of inserting k between i and j
    def get_insertion_cost(self, i, j, k):
        dik = self.dist_matrix[i][k]
        dkj = self.dist_matrix[k][j]
        dij = self.dist_matrix[i][j]
        return dik + dkj - dij

    # solve TSP with RNN method
    def greedy_nearest_neighbor(self, s=None):
        def build_cycle(cluster, start):
            # in cluster, use nn to build a cycle that starts with start
            cycle = [start]
            current = start
            while len(cycle) < len(cluster):
                nn, _ = self.find_nearest_neighbour(current, cluster, cycle)
                cycle.append(nn)
                current = nn
            cycle.append(start)
            return cycle

        def get_shortest_cycle(cluster):
            # find the shortest possible cycle in cluster
            cycles = []
            while len(cycles) < len(cluster):
                cycle = build_cycle(cluster, cluster[len(cycles)])
                cycles.append(cycle)
            cycles_len = [(p, self.get_cycle_length(cycles[p])) for p in range(len(cycles))]
            cycles_len.sort(key=lambda x: x[1])
            shortest_cycle = cycles[cycles_len[0][0]]
            # transform the path so it starts properly
            start = cluster[0]
            if shortest_cycle[0] != start:
                start_id = shortest_cycle.index(start)
                shortest_cycle = shortest_cycle[start_id:-1] + shortest_cycle[:start_id] + [start]
            return shortest_cycle

        # set starting points
        start1, start2 = self.get_starting_points(s=s)
        print("Starting points: ", start1, start2)
        # group vertices
        cluster1, cluster2 = self.group_vertices(start1, start2)
        vis = TSP_visualiser(self.coords, 'Greedy nearest neighbour', cluster1, cluster2)
        # find the shortest path in cluster1 and cluster2
        path1 = get_shortest_cycle(cluster1)
        path2 = get_shortest_cycle(cluster2)
        # get the path length
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1=path1, path2=path2, distance=path_len)

        return path_len

    def greedy_cycle(self, s=None):
        def build_cycle(start, cluster):
            # create a path from start to the nearest vertex
            nn, _ = self.find_nearest_neighbour(start, cluster, [start])
            path = [start, nn, start]
            # while we don't have the full cycle
            while len(path) < len(cluster) + 1:
                # find an unvisited point that is the nearest to any point in the cycle
                k, k_dist = self.find_nearest_neighbour(path[0], cluster, path)
                for point in path[1:-1]:
                    k_tmp, k_dist_tmp = self.find_nearest_neighbour(point, cluster, path)
                    if k_dist_tmp < k_dist:
                        k = k_tmp
                        k_dist = k_dist_tmp
                # in the cycle, look for a vertex {i, j} with the lowest cost of inserting k to the cycle
                i = 0
                insert_cost = self.get_insertion_cost(path[0], path[1], k)
                for i_tmp in range(len(path) - 1):
                    tmp_cost = self.get_insertion_cost(path[i_tmp], path[i_tmp + 1], k)
                    if tmp_cost < insert_cost:
                        i = i_tmp
                        insert_cost = tmp_cost
                # insert k between i and j
                path.insert(i + 1, k)
                # show path on plot
                if path[0] == cluster1[0]:
                    vis.update_graph(path1=path)
                else:
                    vis.update_graph(path2=path)
            return path

        # set starting points
        start1, start2 = self.get_starting_points(s=s)
        # group vertices
        cluster1, cluster2 = self.group_vertices(start1, start2)
        vis = TSP_visualiser(self.coords, 'Greedy cycle', cluster1, cluster2)
        # build paths in cluster1 and cluster2
        path1 = build_cycle(start1, cluster1)
        path2 = build_cycle(start2, cluster2)
        # get path lengths
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1=path1, path2=path2, distance=path_len)
        return path_len

    def furthest_insert(self, s=None):
        def build_cycle(start, cluster):
            # create a path from start to the furthest vertex
            nn, _ = self.find_furthest_neighbour(start, cluster, [start])
            path = [start, nn, start]
            # find k (unvisited point that if the furthest from any point in cycle)
            while len(path) < len(cluster) + 1:
                k, k_dist = self.find_furthest_neighbour(path[0], cluster, path)
                for point in path[1:-1]:
                    k_tmp, k_dist_tmp = self.find_furthest_neighbour(point, cluster, path)
                    if k_dist_tmp < k_dist:
                        k = k_tmp
                        k_dist = k_dist_tmp
                # in the cycle, look for a vertex {i, j} with the lowest cost of inserting k to cycle
                i = 0
                insert_cost = self.get_insertion_cost(path[0], path[1], k)
                for i_tmp in range(len(path) - 1):
                    tmp_cost = self.get_insertion_cost(path[i_tmp], path[i_tmp + 1], k)
                    if tmp_cost < insert_cost:
                        i = i_tmp
                        insert_cost = tmp_cost
                # insert k between i and j
                path.insert(i + 1, k)

                # show path on plot
                if path[0] == cluster1[0]:
                    vis.update_graph(path1=path)
                else:
                    vis.update_graph(path2=path)
            return path

        # set starting points
        start1, start2 = self.get_starting_points(s=s)
        # group vertices
        cluster1, cluster2 = self.group_vertices(start1, start2)
        vis = TSP_visualiser(self.coords, 'Furthest insert', cluster1, cluster2)
        # build cycles for start1 and start2
        path1 = build_cycle(start1, cluster1)
        path2 = build_cycle(start2, cluster2)
        # get path length
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1=path1, path2=path2, distance=path_len)
        return path_len

    def regret(self, s=None):
        def get_point_with_biggest_regret(path, cluster):
            # collect costs
            all_costs = []
            neighbors = []
            for point in cluster:
                # for every unvisited point in cluster
                if point not in path:
                    neighbors.append(point)
                    costs = []
                    for i in range(len(path) - 1):
                        insert_cost = self.get_insertion_cost(path[i], path[i + 1], point)
                        costs.append((i, insert_cost))
                    costs.sort(key=lambda x: x[1])
                    all_costs.append(costs)
            # calculate 2-regret
            regrets = []
            for i in range(len(all_costs)):
                point = all_costs[i]
                regret = point[1][1] - point[0][1]
                regrets.append((neighbors[i], regret))
            # sort points by the biggest regret
            regrets.sort(key=lambda x: x[1], reverse=True)
            point_to_add = regrets[0][0]
            return point_to_add, all_costs[neighbors.index(point_to_add)][0][0]

        def build_cycle(start, cluster):
            # create a path from start to the nearest vertex
            nn, _ = self.find_nearest_neighbour(start, cluster, [start])
            path = [start, nn, start]
            # while we don't have the full cycle, insert point with the biggest regret
            while len(path) < len(cluster) + 1:
                p, i = get_point_with_biggest_regret(path, cluster)
                path.insert(i + 1, p)
                # draw path on plot
                if start == cluster1[0]:
                    vis.update_graph(path1=path)
                else:
                    vis.update_graph(path2=path)
            return path

        # set starting points
        start1, start2 = self.get_starting_points(s=s)
        # group vertices
        cluster1, cluster2 = self.group_vertices(start1, start2)
        vis = TSP_visualiser(self.coords, '2-regret', cluster1, cluster2)
        # build cycles for start1 and start2
        path1 = build_cycle(start1, cluster1)
        path2 = build_cycle(start2, cluster2)
        # get path length
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1, path2, distance=path_len)
        return path_len

    def algo_test(self, filename):
        f = open(filename, "w")
        for x in range(100):
            p_len = self.regret(s=x)
            print("test", x, "len:", p_len)
            f.write(str(p_len) + '\n')
        f.close()


solver = TSP_solver(kroA100_filename)
# solver.algo_test('2r_test_B.txt')
solver.greedy_cycle()
