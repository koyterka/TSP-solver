import os
import numpy as np
import math
from TSP_visualiser import TSP_visualiser
from random import randrange
import random
from random import shuffle
import time
import itertools
from operator import itemgetter

kroA100_filename = 'kroA100.tsp'
kroB100_filename = 'kroB100.tsp'
kroA200_filename = 'kroA200.tsp'
kroB200_filename = 'kroB200.tsp'

test_coords = [[2, 5], [3, 7], [5, 8], [7.5, 6.5], [9, 5.5], [6.5, 4], [7.5, 9], [9, 12], [13, 12], [14, 9],
               [12.5, 9.5],
               [10.5, 10.5], [10.5, 8]]


class TSP_solver:
    def __init__(self, filename):
        self.coords = self.get_graph_from_tsp(filename)
        #self.coords = test_coords
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
        if cycle is None:
            return cycle_len
        else:
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

    def is_edge_in_path(self, edge, path):
        # if edge[0] in path and edge[1] in path:
        #     if path.index(edge[0]) != len(path)-1 and path[path.index(edge[0]) + 1] == edge[1]:
        #         return edge
        #     else:
        #         edge.reverse()
        #         if path.index(edge[0]) != len(path) - 1 and path[path.index(edge[0]) + 1] == edge[1]:
        #             return edge
        #
        # # elif path.index(edge[1])!=len(path)-1 and edge[1] in path and path[path.index(edge[1]) + 1] == edge[0]:
        # #     return True
        # else:
        #     return False
        n1, n2 = self.get_vertices_nearby(path, edge[0])
        if n1 is not None and n2 is not None:
            if [n1, edge[0]] == edge or [edge[0], n2] == edge:
                return True
        return False

    def find_path(self, v1, path1):
        return 0 if v1 in path1 else 1

    def get_edge_cost(self, v1, v2):
        return self.dist_matrix[v1][v2]

    def del_common_from_neigh(self, n1, n2):
        return [n for n in n1 if n not in n2]

    def get_delta(self, move, path1, path2):
        deleted_edges, added_edges = 0, 0
        if len(move) > 1:  # edges swap
            deleted_edges += self.dist_matrix[move[0][0]][move[0][1]]
            deleted_edges += self.dist_matrix[move[1][0]][move[1][1]]
            added_edges += self.dist_matrix[move[0][0]][move[1][0]]
            added_edges += self.dist_matrix[move[0][1]][move[1][1]]
        else:  # vertex swap
            paths = [path1, path2]
            v1, v2 = move[0][0], move[0][1]

            v1path, v2path = self.find_path(v1, path1), self.find_path(v2, path1)
            n11, n12 = self.get_vertices_nearby(paths[v1path], v1)
            n21, n22 = self.get_vertices_nearby(paths[v2path], v2)
            deleted_edges += self.dist_matrix[n11][v1]
            deleted_edges += self.dist_matrix[v1][n12]
            deleted_edges += self.dist_matrix[n21][v2]
            deleted_edges += self.dist_matrix[v2][n22]
            added_edges += self.dist_matrix[n11][v2]
            added_edges += self.dist_matrix[v2][n12]
            added_edges += self.dist_matrix[n21][v1]
            added_edges += self.dist_matrix[n22][v1]

            #n1, n2 = [n11, n12], [n21, n22]
            # if v1path == v2path:
            #     n1 = self.del_common_from_neigh([n11, n12], [n21, v2, n22])
            #     n2 = self.del_common_from_neigh([n21, n22], [n11, v1, n12])
            #
            # for n in n1:
            #     deleted_edges += self.get_edge_cost(n, v1)
            #     added_edges += self.get_edge_cost(n, v2)
            # for n in n2:
            #     deleted_edges += self.get_edge_cost(n, v2)
            #     added_edges += self.get_edge_cost(n, v1)

        return added_edges - deleted_edges

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

    def get_point_with_biggest_regret(self, path, cluster):
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

    def regret(self, s=None):
        def build_cycle(start, cluster):
            # create a path from start to the nearest vertex
            nn, _ = self.find_nearest_neighbour(start, cluster, [start])
            path = [start, nn, start]
            # while we don't have the full cycle, insert point with the biggest regret
            while len(path) < len(cluster) + 1:
                p, i = self.get_point_with_biggest_regret(path, cluster)
                path.insert(i + 1, p)
                # draw path on plot
                # if start == cluster1[0]:
                #     vis.update_graph(path1=path)
                # else:
                #     vis.update_graph(path2=path)
            return path

        # set starting points
        start1, start2 = self.get_starting_points(s=s)
        # group vertices
        cluster1, cluster2 = self.group_vertices(start1, start2)
        # vis = TSP_visualiser(self.coords, '2-regret', cluster1, cluster2)
        # build cycles for start1 and start2
        path1 = build_cycle(start1, cluster1)
        path2 = build_cycle(start2, cluster2)
        # get path length
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        # vis.keep_graph()
        # vis.update_graph(path1, path2, distance=path_len)
        return path1, path2, path_len, start2

    def get_vertices_nearby(self, path, v):
        n1, n2 = None, None
        if v in path:
            n1 = path[path.index(v) - 1]
            n2 = (
                path[path.index(v) + 1] if (path.index(v) + 1) < len(path) else path[0]
            )
        return n1, n2

    def get_vertices_swap_neighbourhood(self, path1, path2=None):
        N = []
        if path2:
            for m in list(itertools.product(path1[:-1], path2[:-1])):
                N.append([m])
        else:
            for i1, e1 in enumerate(path1[:-1]):
                for e2 in path1[i1 + 1:-1]:
                    N.append([(e1, e2)])
        return N

    def get_edges_swap_neighbourhood(self, path):
        e = [[path[d - 1], vertex] for d, vertex in enumerate(path)]
        return [
            item
            for item in itertools.combinations(e, 2)
            if len(set(itertools.chain(*item))) == 4
        ]
        # N = []
        # for i1, v1 in enumerate(path[:-2]):
        #     for i2 in range(len(path[i1 + 1:-1])):
        #         if path[i1 + 1 + i2] not in [path[i1], path[i1 + 1]] and path[i2 + i1 + 2] not in [path[i1],
        #                                                                                            path[i1 + 1]]:
        #             N.append([[path[i1], path[i1 + 1]], [path[i2 + i1 + 1], path[i2 + i1 + 2]]])
        # return N

    def apply_move_to_path(self, path1, path2, move):
        # edges swap [[4,5],[7,8]]
        if len(move) > 1:
            first = move[0][1] in path1
            modified_path = path1 if first else path2
            if modified_path.index(move[0][0]) > modified_path.index(move[1][0]):
                move = [move[1], move[0]]
            # move[0] = self.is_edge_in_path(move[0], modified_path)
            # move[1] = self.is_edge_in_path(move[1], modified_path)

            # if move[0] and move[1]:
            # e1 = move[0][1]  # 5
            # e2 = move[1][0]  # 7
            id1 = modified_path.index(move[0][1])
            id2 = modified_path.index(move[1][0])

            modified_path[id1], modified_path[id2] = modified_path[id2], modified_path[id1]

            if id1 < id2:
                modified_path[id1 + 1:id2] = modified_path[id1 + 1:id2][::-1]
            else:
                modified_path[id2 + 1:id1] = modified_path[id2 + 1:id1][::-1]

            return [modified_path, path2] if first else [path1, modified_path]

        # vertices swap
        else:
            paths_to_return = [path1, path2]
            e1, e2 = move[0][0], move[0][1]

            is1, is2 = int(e1 not in path1), int(e2 not in path1)
            # 1 1 -- both in path2
            # 0 0 -- both in path1
            # 1 0 -- e1 in path2, e2 in path1
            # 0 1 -- e1 in path1, e2 in path2

            id1, id2 = paths_to_return[is1].index(e1), paths_to_return[is2].index(e2)
            paths_to_return[is1][id1], paths_to_return[is2][id2] = paths_to_return[is2][id2], paths_to_return[is1][id1]

            for i, path in enumerate(paths_to_return):
                paths_to_return[i] = path[:-1]
                paths_to_return[i].append(path[0])

            return paths_to_return[0], paths_to_return[1]

    def get_neighbourhood(self, path1, path2=None, mode='e'):
        # get outer vertices neighbourhood
        N = self.get_vertices_swap_neighbourhood(path1=path1, path2=path2)
        #get inner vertices neighbourhood
        if mode == 'v':
            N2 = self.get_vertices_swap_neighbourhood(path1=path1, path2=None)
            N3 = self.get_vertices_swap_neighbourhood(path1=path2, path2=None)
        # get inner edges neighbourhood
        else:
            N2 = self.get_edges_swap_neighbourhood(path=path1)
            N3 = self.get_edges_swap_neighbourhood(path=path2)
        N = N + N2 + N3
        return N

    def get_best_move(self, N, path1, path2):
        best_delta = np.inf
        best_move = None
        for m in N:
            dm = self.get_delta(m, path1, path2)
            if dm < best_delta:
                best_delta = dm
                best_move = m
        return best_move

    def randomize_neighbourhood(self, N):
        n = random.choice(range(len(N)))
        return N[n:] + N[:n]

    def steepest(self, path1, path2=None, mode='e', show=False):
        if show:
            cluster1 = [x for x in range(len(self.dist_matrix)) if x in path1]
            cluster2 = [x for x in range(len(self.dist_matrix)) if x not in path1]
            vis = TSP_visualiser(self.coords, 'Steepest ' + mode, cluster1, cluster2)

        found = True
        while found:
            found = False
            N = self.get_neighbourhood(path1, path2, mode)
            m = self.get_best_move(N, path1, path2)
            if self.get_delta(m, path1, path2) < 0:
                [path1, path2] = self.apply_move_to_path(path1, path2, m)
                if show:
                    vis.update_graph(path1=path1, path2=path2, distance=0)
                found = True

        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)

        if show:
            vis.keep_graph()
            vis.update_graph(path1=path1, path2=path2, distance=path_len)

        return path1, path2, path_len

    def greedy(self, path1, path2=None, mode='e'):
        cluster1 = [x for x in range(len(self.dist_matrix)) if x in path1]
        cluster2 = [x for x in range(len(self.dist_matrix)) if x in path2]
        vis = TSP_visualiser(self.coords, 'Greedy ' + mode, cluster1, cluster2)

        original_path1 = path1.copy()
        original_path2 = path2.copy()

        found = True
        while found:
            found = False
            N = self.get_neighbourhood(path1, path2, mode)
            randomized_N = self.randomize_neighbourhood(N)
            for m in randomized_N:
                if self.get_delta(m, path1, path2) < 0:
                    [path1, path2] = self.apply_move_to_path(path1, path2, m)
                    vis.update_graph(path1=path1, path2=path2, distance=0)
                    found = True
                    break

        improv = self.get_cycle_length(original_path1) + self.get_cycle_length(original_path2) - self.get_cycle_length(
            path1) - self.get_cycle_length(path2)
        percent = improv / (self.get_cycle_length(original_path1) + self.get_cycle_length(original_path2))
        print("Improvement:", improv)
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1=path1, path2=path2, distance=path_len)
        return path1, path2, improv, percent

    def build_LM(self, path1, path2):
        LM = []
        N = self.get_neighbourhood(path1=path1, path2=path2, mode='e')
        for move in N:
            delta = self.get_delta(move, path1, path2)
            LM.append((move, delta))
        return LM

    # sort LM from best to worst move (best=smallest delta)
    def sort_LM(self, LM):
        return sorted(LM, key=itemgetter(1), reverse=False)

    def move_is_applicable(self, move, path1, path2):
        if len(move) > 1:  # edges swap
            # check if edges are still in path
            e1, e2 = move[0], move[1]
            if self.is_edge_in_path(e1, path1) and self.is_edge_in_path(e2, path1):
                return True
            elif self.is_edge_in_path(e1, path2) and self.is_edge_in_path(e2, path2):
                return True
            else:
                return False

        else:  # vertex swap
            # check if vertices are still in opposite paths
            v1, v2 = move[0][0], move[0][1]
            v1path, v2path = self.find_path(v1, path1), self.find_path(v2, path1)
            # vertices are in the same path
            if v1path == v2path:
                return False
            else:
                return True

    def last_moves_evaluation(self, path1, path2, show=False):
        def get_impacted_vertices(move):
            if len(move) > 1:  # edge change
                # impacted = [v for v in move[0]]
                # impacted.extend(move[1])
                return [item for sublist in move for item in sublist]
            else:  # vertex change
                if move[0][0] in path1:
                    n1, n2 = self.get_vertices_nearby(path1, move[0][0])
                    impacted = [n1, n2]
                    impacted.extend(self.get_vertices_nearby(path2, move[0][1]))
                else:
                    n1, n2 = self.get_vertices_nearby(path2, move[0][0])
                    impacted = [n1, n2]
                    impacted.extend(self.get_vertices_nearby(path1, move[0][1]))
                #impacted.extend([move[0][0], move[0][1]])
                impacted += move[0]
                return impacted

        # do this before applying move
        def get_impacted_moves(LM, move):
            impacted = get_impacted_vertices(eval(move))
            to_delete = []
            for k in LM:
                m = eval(k)
                if len(m) == 1 and (m[0][0] in impacted or m[0][1] in impacted):
                    to_delete.append(k)
                elif len(m) > 1:
                    if not set(m[0]+m[1]).isdisjoint(impacted):
                        # or not set(m[1]).isdisjoint(impacted)
                        to_delete.append(k)
            return to_delete

        def update_LM(LM, N):
            new_LM = {}
            for move in N:
                key = str(move)
                if key not in LM.keys():
                    delta = self.get_delta(move, path1, path2)
                    if delta < 0:
                        #LM[key] = delta
                        new_LM[key] = delta
                else:
                    new_LM[key] = LM[key]
            # sort moves in LM from best to worst
            #return dict(sorted(LM.items(), key=lambda item: item[1]))
            return dict(sorted(new_LM.items(), key=lambda item: item[1]))

        if show:
            cluster1 = [x for x in range(len(self.coords)) if x in path1]
            cluster2 = [x for x in range(len(self.coords)) if x not in path1]
            vis = TSP_visualiser(self.coords, 'Last-iteration', cluster1, cluster2)

        # dict of moves that improve the solution, sorted from best to worst
        LM = {}

        found = True
        while found:
            found = False

            # get new moves
            N = self.get_neighbourhood(path1=path1, path2=path2, mode='e')

            # update LM
            LM = update_LM(LM, N)

            to_delete = []
            for move in LM:
                if self.move_is_applicable(eval(move), path1, path2):
                    #print("\n\napplying move:", move, "with delta", LM[move])
                    found = True
                    imp = get_impacted_moves(LM, move)
                    to_delete.extend(imp)
                    [path1, path2] = self.apply_move_to_path(path1, path2, eval(move))
                    if show:
                        vis.update_graph(path1, path2)
                    break
                else:
                    to_delete.append(move)

            to_delete = np.unique(to_delete)
            for move_to_del in to_delete:
                LM.pop(move_to_del)

        if show:
            full_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
            vis.keep_graph()
            vis.update_graph(path1, path2, distance=full_len)

        full_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        return path1, path2, full_len

    def get_n_nearest_neighbors(self, vertex, n=10):
        all_neighs = [(i, d) for i, d in enumerate(self.dist_matrix[vertex]) if d > 0]
        all_neighs = sorted(all_neighs, key=itemgetter(1))
        if len(all_neighs) >= n:
            return [i for (i, d) in all_neighs[:n]]
        else:
            return [i for (i, d) in all_neighs]

    def get_resulting_edges(self, path1, path2, move):
        if len(move) > 1:  # edge swap
            return [[move[0][0], move[1][0]], [move[0][1], move[1][1]]]
        else:  # vertex swap
            paths = [path1, path2]
            v1, v2 = move[0][0], move[0][1]
            #v1path, v2path = self.find_path(v1, path1), self.find_path(v2, path1)
            v1path, v2path = int(v1 not in path1), int(v2 not in path2)
            n11, n12 = self.get_vertices_nearby(paths[v1path], v1)
            n21, n22 = self.get_vertices_nearby(paths[v2path], v2)
            return [[n11, v2], [v2, n12], [n21, v1], [n22, v1]]

    def get_cycles_len_sum(self, path1, path2):
        return self.get_cycle_length(path1) + self.get_cycle_length(path2)

    def map_paths(self, p1, p2):
        N_vertices = len(self.dist_matrix)
        # vmap = []
        # for v in range(len(self.dist_matrix)):
        #     if v in p1:
        #         vmap.append(p1.index(v))
        #     else:
        #         vmap.append(p2.index(v))
        return [(0, p1.index(v)) if v in p1 else (1, p2.index(v)) for v in range(N_vertices)]

    def get_all_n_neighbors(self, n_neighbors):
        N_vertices = len(self.dist_matrix)
        return [self.get_n_nearest_neighbors(v, n_neighbors) for v in range(N_vertices)]

    def candidate_moves_evaluation(self, path1, path2, n_neighs=2, v_range=None, show=False):
        if show:
            print("initializing vis...")
            cluster1 = [x for x in range(len(self.coords)) if x in path1]
            cluster2 = [x for x in range(len(self.coords)) if x not in path1]
            vis = TSP_visualiser(self.coords, 'Candidate moves', cluster1, cluster2)
            vis.update_graph(path1, path2)

        v_neighbors = self.get_all_n_neighbors(n_neighs)
        
        if v_range is None:
            v_range = [v for v in range(len(self.dist_matrix))]

        paths = [path1, path2]
        N = []
        v_map = self.map_paths(path1, path2)

        for v1 in v_range:
            v1n1, v1n2 = self.get_vertices_nearby(paths[v_map[v1][0]], v1)

            for v2 in v_neighbors[v1]:
                v2n1, v2n2 = self.get_vertices_nearby(paths[v_map[v2][0]], v2)
                if v_map[v1][0] == v_map[v2][0]:
                    # edges swap
                    N.append([[v1, v1n2], [v2, v2n2]])
                    N.append([[v1n1, v1], [v2n1, v2]])
                else:
                    # vertices swap
                    N.append([(v1n1, v2)])
                    N.append([(v1n2, v2)])
                    N.append([(v1, v2n1)])
                    N.append([(v1, v2n2)])

            LM = [(move, self.get_delta(move, path1, path2)) for move in N]
            LM = self.sort_LM(LM)
            best_move, delta = LM[0][0], LM[0][1]

            if delta < 0:
                [path1, path2] = self.apply_move_to_path(path1, path2, best_move)
                paths = [path1, path2]
                N = []
                v_map = self.map_paths(path1, path2)

                if show:
                    distance = self.get_cycle_length(path1) + self.get_cycle_length(path2)
                    vis.update_graph(path1, path2, distance)

        distance = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        if show:
            vis.keep_graph()
            vis.update_graph(path1, path2, distance=distance)

        return path1, path2, distance

    def perturbation1(self, path1, path2, no_swaps=6):
        if no_swaps >= 3:
            swaps = [no_swaps // 3 + (1 if x < no_swaps % 3 else 0) for x in range(3)]

            for s in range(swaps[0]):
                e1 = self.get_edges_swap_neighbourhood(path1)
                move = random.choice(e1)
                [path1, path2] = self.apply_move_to_path(path1, path2, move)

            for s in range(swaps[1]):
                e2 = self.get_edges_swap_neighbourhood(path2)
                move = random.choice(e2)
                [path1, path2] = self.apply_move_to_path(path1, path2, move)

            for s in range(swaps[2]):
                v1, v2 = random.choice(path1), random.choice(path2)
                move = [(v1, v2)]
                [path1, path2] = self.apply_move_to_path(path1, path2, move)

        return path1, path2

    def perturbation2(self, path1, path2, percentage=20):
        if percentage <= 0 or percentage > 100:
            percentage = 20
        part = percentage/100
        no_to_destroy = int(len(self.dist_matrix)*part)
        free_vertices = []

        def destroy_random_vertices(path):
            path = path[:-1]
            vertices = path.copy()
            for d in range(int(no_to_destroy/2)):
                v = random.choice(vertices)
                vertices.remove(v)
                free_vertices.append(v)
                path[path.index(v)] = None
            return path

        def destroy_longest_edges(path):
            edges = [[vertex, path[d+1]] for d, vertex in enumerate(path[:-1])]
            costs = [(i, self.get_edge_cost(edge[0], edge[1])) for i, edge in enumerate(edges)]
            costs.sort(key=lambda x: x[1], reverse=True)
            costs = costs[:int(no_to_destroy/2)]

            vertices_to_delete = []
            for (i, _) in costs:
                [v1, v2] = edges[i]
                vertices_to_delete.append(v1)
                vertices_to_delete.append(v2)

            vertices_to_delete = np.unique(vertices_to_delete)
            for v in vertices_to_delete:
                free_vertices.append(v)
                path[path.index(v)] = None
            return path

        def greedy_cycle_repair(path):
            desired_path_len = len(self.dist_matrix) / 2
            # while we don't have the full cycle
            while len(path) < desired_path_len + 1:
                # find an unvisited point that is the nearest to any point in the cycle
                k, k_dist = self.find_nearest_neighbour(path[0], free_vertices, path)
                for point in path[1:-1]:
                    k_tmp, k_dist_tmp = self.find_nearest_neighbour(point, free_vertices, path)
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
                free_vertices.remove(k)
            return path

        def repair(path):
            # delete all Nones from path
            path = [x for x in path if x is not None]
            #path[-1] = path[0]
            if path[0] != path[-1]:
                path.append(path[0])
            # repair path
            path = greedy_cycle_repair(path)
            return path

        path1, path2 = destroy_random_vertices(path1), destroy_random_vertices(path2)
        path1, path2 = repair(path1), repair(path2)
        return path1, path2

    def MSLS(self, path1, path2, n_neighs=7, LS_iterations=100):
        vertices_range = [v for v in range(len(self.dist_matrix))]
        solutions = []

        start = time.time()
        for it in range(LS_iterations):
            p1, p2 = path1.copy(), path2.copy()
            shuffle(vertices_range)
            p1, p2, dist = self.candidate_moves_evaluation(p1, p2, n_neighs, v_range=vertices_range)
            #print("got:", dist)
            solutions.append(([p1, p2], dist))
        end = time.time()
        elapsed = end-start
        solutions = self.sort_LM(solutions)
        print("best:", solutions[0][1])
        [p1, p2], dist = solutions[0][0], solutions[0][1]
        return p1, p2, dist, elapsed

    def ILS1(self, path1, path2, end_time=4.5, n_neighs=7):
        x1, x2, _ = self.candidate_moves_evaluation(path1, path2, n_neighs)
        best_dist = self.get_cycles_len_sum(x1, x2)

        timeout = time.time() + end_time
        while True:
            # perturbacja
            y1, y2 = x1.copy(), x2.copy()
            y1, y2 = self.perturbation1(y1, y2, no_swaps=6)
            # local search
            y1, y2, _ = self.candidate_moves_evaluation(y1, y2, n_neighs=n_neighs)
            # jeśli f(y)>f(x)
            len_before = self.get_cycles_len_sum(x1, x2)
            len_after = self.get_cycles_len_sum(y1, y2)
            #print("got", len_after, "compare to", len_before)
            if len_after < len_before:
                #print(len_after, "is better!")
                x1, x2 = y1.copy(), y2.copy()
                best_dist = len_after
            # warunek stopu
            if time.time() > timeout:
                break
        return x1, x2, best_dist

    def ILS2(self, path1, path2, end_time=5, n_neighs=10, percentage=20, localsearch=True):
        x1, x2, _ = self.candidate_moves_evaluation(path1, path2, n_neighs)
        best_dist = self.get_cycles_len_sum(x1, x2)

        timeout = time.time() + end_time
        while True:
            # perturbacja
            y1, y2 = x1.copy(), x2.copy()
            y1, y2 = self.perturbation2(y1, y2, percentage=percentage)

            if localsearch:
                # local search
                y1, y2, _ = self.candidate_moves_evaluation(y1, y2, n_neighs=n_neighs)

            # jeśli f(y)>f(x)
            len_before = self.get_cycles_len_sum(x1, x2)
            len_after = self.get_cycles_len_sum(y1, y2)
            #print("got", len_after, "compared to", len_before)
            if len_after < len_before:
                #print(len_after, "is better!")
                x1, x2 = y1.copy(), y2.copy()
                best_dist = len_after
            # warunek stopu
            if time.time() > timeout:
                break
        return x1, x2, best_dist

    def generate_random_paths(self):
        ran = [i for i in range(len(self.dist_matrix))]
        split = int(len(self.dist_matrix) / 2)

        a_random_path = random.sample(ran, split)
        a_random_path.append(a_random_path[0])
        b_random_path = [i for i in ran if i not in a_random_path]
        shuffle(b_random_path)
        b_random_path.append(b_random_path[0])

        # print("\n\nPATHS:", a_random_path, b_random_path)
        return a_random_path, b_random_path

    def heuristic_algo_test(self, filename, time_filename):
        f = open(filename, "w")
        time_f = open(time_filename, "w")
        for x in range(100):
            start = time.time()
            _, _, p_len, s2 = self.regret(s=x)
            end = time.time()
            t = end - start
            print("test", x, s2, "len:", p_len, "time:", t)
            f.write(str(p_len) + '\n')
            time_f.write(str(t) + '\n')
        f.close()
        time_f.close()

    def construction_test(self, filename1, filename2):
        f1 = open(filename1, "w")
        f2 = open(filename2, "w")
        best_before_1, best_before_2, best_after_1, best_after_2 = None, None, None, None
        best_len = np.inf
        no_of_tests = 10
        for x in range(no_of_tests):
            s1 = random.choice([x for x in range(len(self.dist_matrix))])
            p1, p2 = self.get_random_paths_clustered(s1=s1)
            start = time.time()
            print("building...")
            #p1after, p2after, path_len, elapsed = self.MSLS(path1=p1, path2=p2, n_neighs=10)
            p1after, p2after, path_len = self.ILS2(p1, p2, end_time=5.5, n_neighs=10, localsearch=True)
            #p1after, p2after, path_len = self.candidate_moves_evaluation(p1, p2, n_neighs=10)
            end = time.time()
            elapsed = end - start
            print("test", x, "len:", path_len, "time:", elapsed)

            if path_len < best_len:
                best_before_1, best_before_2 = p1, p2
                best_after_1, best_after_2 = p1after, p2after
                best_len = path_len
                print("best", path_len)

            f1.write(str(path_len) + '\n')
            f2.write(str(elapsed) + '\n')
        f1.close()
        f2.close()

        self.draw_path(best_after_1, best_after_2)
        return best_before_1, best_before_2, best_after_1, best_after_2

    def get_random_paths_clustered(self, s1):
        #s1, s2 = self.get_starting_points(s)
        # s2 = s1
        # while s1 == s2:
        #     s2 = random.randint(0, len(self.dist_matrix)-1)
        distances = self.dist_matrix[s1]
        s2 = distances.index(max(distances))
        cluster1, cluster2 = self.group_vertices(s1, s2)
        random.shuffle(cluster1)
        random.shuffle(cluster2)
        cluster1.append(cluster1[0])
        cluster2.append(cluster2[0])
        return cluster1, cluster2


    def draw_path(self, p1, p2):
        cluster1 = [x for x in range(len(self.dist_matrix)) if x in p1]
        cluster2 = [x for x in range(len(self.dist_matrix)) if x in p2]
        vis = TSP_visualiser(self.coords, 'Best of ILS2a, kroB200', cluster1, cluster2)
        vis.keep_graph()
        dist = self.get_cycle_length(p1) + self.get_cycle_length(p2)
        vis.update_graph(path1=p1, path2=p2, distance=dist)


solver = TSP_solver(kroA200_filename)
solver.construction_test("test_A.txt", "test_time_A.txt")
