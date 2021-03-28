import os
import numpy as np
import math
from TSP_visualiser import TSP_visualiser
from random import randrange
import random
from random import shuffle
import time
import itertools
kroA100_filename = 'kroA100.tsp'
kroB100_filename = 'kroB100.tsp'

test_coords = [[2,5], [3,7], [5,8], [7.5, 6.5], [9,5.5], [6.5, 4], [7.5, 9], [9,12], [13, 12], [14,9], [12.5, 9.5],
               [10.5, 10.5], [10.5, 8], [7.5, 9]]

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

    def find_path(self, v1, path1):
        return 0 if v1 in path1 else 1

    def get_edge_cost(self, v1, v2):
        return self.dist_matrix[v1][v2]

    def del_common_from_neigh(self, n1, n2):
        return [n for n in n1 if n not in n2]

    def get_delta(self, move, path1, path2):
        deleted_edges, added_edges = 0, 0
        if len(move) > 1:   # edges swap
            deleted_edges += self.dist_matrix[move[0][0]][move[0][1]]
            deleted_edges += self.dist_matrix[move[1][0]][move[1][1]]
            added_edges += self.dist_matrix[move[0][0]][move[1][0]]
            added_edges += self.dist_matrix[move[0][1]][move[1][1]]
        else:   # vertex swap
            paths = [path1, path2]
            v1, v2 = move[0][0], move[0][1]
            v1path, v2path = self.find_path(v1, path1), self.find_path(v2, path1)
            n11, n12 = self.get_vertices_nearby(paths[v1path], v1)
            n21, n22 = self.get_vertices_nearby(paths[v2path], v2)
            n1, n2 = [n11, n12], [n21, n22]

            if v1path == v2path:
                n1 = self.del_common_from_neigh([n11, n12], [n21, v2, n22])
                n2 = self.del_common_from_neigh([n21, n22], [n11, v1, n12])

            for n in n1:
                deleted_edges += self.get_edge_cost(n, v1)
                added_edges += self.get_edge_cost(n, v2)
            for n in n2:
                deleted_edges += self.get_edge_cost(n, v2)
                added_edges += self.get_edge_cost(n, v1)

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
                # if start == cluster1[0]:
                #     vis.update_graph(path1=path)
                # else:
                #     vis.update_graph(path2=path)
            return path

        # set starting points
        start1, start2 = self.get_starting_points(s=s)
        # group vertices
        cluster1, cluster2 = self.group_vertices(start1, start2)
        #vis = TSP_visualiser(self.coords, '2-regret', cluster1, cluster2)
        # build cycles for start1 and start2
        path1 = build_cycle(start1, cluster1)
        path2 = build_cycle(start2, cluster2)
        # get path length
        #path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        # vis.keep_graph()
        # vis.update_graph(path1, path2, distance=path_len)
        return path1, path2

    def get_vertices_nearby(self, path, v):
        id = path.index(v)
        if id == 0 or id == len(path)-1:
            return [path[len(path)-2], path[1]]
        else:
            return path[id-1], path[id+1]

    def get_vertices_swap_neighbourhood(self, path1, path2=None):
        N = []
        if path2:
            for m in list(itertools.product(path1[:-1], path2[:-1])):
                N.append([m])
        else:
            for i1, e1 in enumerate(path1[:-1]):
                for e2 in path1[i1+1:-1]:
                    N.append([(e1, e2)])
        return N

    def get_edges_swap_neighbourhood(self, path):
        N = []
        for i1, v1 in enumerate(path[:-2]):
            for i2 in range(len(path[i1 + 1:-1])):
                if path[i1 + 1 + i2] not in [path[i1], path[i1 + 1]] and path[i2 + i1 + 2] not in [path[i1],
                                                                                                   path[i1 + 1]]:
                    N.append([[path[i1], path[i1 + 1]], [path[i2 + i1 + 1], path[i2 + i1 + 2]]])
        return N

    def apply_move_to_path(self, path1, path2, move):
        # edges swap [[4,5],[7,8]]
        if len(move) > 1:
            e1 = move[0][1] #5
            e2 = move[1][0] #7
            first = e1 in path1
            modified_path = path1.copy() if first else path2.copy()
            id1 = modified_path.index(e1)
            id2 = modified_path.index(e2)
            modified_path[id1], modified_path[id2] = modified_path[id2], modified_path[id1]
            modified_path[id1 + 1:id2] = modified_path[id1+1:id2][::-1]

            return (modified_path, path2) if first else (path1, modified_path)

        # vertices swap
        else:
            paths_to_return = [path1.copy(), path2.copy()]
            e1, e2 = move[0][0], move[0][1]

            is1, is2 = int(e1 not in path1), int(e2 not in path1)
            # 1 1 -- both in path2
            # 0 0 -- both in path1
            # 1 0 -- e1 in path2, e2 in path1
            # 0 1 -- e1 in path1, e2 in path2

            id1, id2 = paths_to_return[is1].index(e1), paths_to_return[is2].index(e2)
            paths_to_return[is1][id1], paths_to_return[is2][id2] = paths_to_return[is2][id2], paths_to_return[is1][id1]

            for i in range(len(paths_to_return)):
                p = paths_to_return[i]
                paths_to_return[i] = p[:-1]
                paths_to_return[i].append(p[0])

            return paths_to_return[0], paths_to_return[1]

    def get_neighbourhood(self, path1, path2=None, mode='e'):
        # get outer vertices neighbourhood
        N = self.get_vertices_swap_neighbourhood(path1=path1, path2=path2)
        # get inner vertices neighbourhood
        if mode == 'v':
            N2 = self.get_vertices_swap_neighbourhood(path1=path1, path2=None)
            N3 = self.get_vertices_swap_neighbourhood(path1=path2, path2=None)
        # get inner edges neighbourhood
        else:
            N2 = self.get_edges_swap_neighbourhood(path=path1)
            N3 = self.get_edges_swap_neighbourhood(path=path2)
        N = N+N2+N3
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
        return N[n:]+N[:n]

    def steepest(self, path1, path2=None, mode='e'):
        cluster1 = [x for x in range(100) if x in path1]
        cluster2 = [x for x in range(100) if x not in path1]
        vis = TSP_visualiser(self.coords, 'Steepest '+mode, cluster1, cluster2)
        original_path1 = path1.copy()
        original_path2 = path2.copy()

        found = True
        while found:
            found = False
            N = self.get_neighbourhood(path1, path2, mode)
            m = self.get_best_move(N, path1, path2)
            if self.get_delta(m, path1, path2) < 0:
                (path1, path2) = self.apply_move_to_path(path1, path2, m)
                vis.update_graph(path1=path1, path2=path2, distance=0)
                found = True

        improv = self.get_cycle_length(original_path1) + self.get_cycle_length(original_path2) - self.get_cycle_length(
            path1) - self.get_cycle_length(path2)
        percent = improv / (self.get_cycle_length(original_path1) + self.get_cycle_length(original_path2))
        print("Improvement:", improv)
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1=path1, path2=path2, distance=path_len)
        return path1, path2, improv, percent

    def greedy(self, path1, path2=None, mode = 'e'):
        cluster1 = [x for x in range(len(self.dist_matrix)) if x in path1]
        cluster2 = [x for x in range(len(self.dist_matrix)) if x in path2]
        vis = TSP_visualiser(self.coords, 'Greedy '+mode, cluster1, cluster2)

        original_path1 = path1.copy()
        original_path2 = path2.copy()

        found = True
        while found:
            found = False
            N = self.get_neighbourhood(path1, path2, mode)
            randomized_N = self.randomize_neighbourhood(N)
            for m in randomized_N:
                if self.get_delta(m, path1, path2) < 0:
                    (path1, path2) = self.apply_move_to_path(path1, path2, m)
                    vis.update_graph(path1=path1, path2=path2, distance=0)
                    found = True
                    break

        improv = self.get_cycle_length(original_path1) + self.get_cycle_length(original_path2) - self.get_cycle_length(path1) - self.get_cycle_length(path2)
        percent = improv/(self.get_cycle_length(original_path1) + self.get_cycle_length(original_path2))
        print("Improvement:", improv)
        path_len = self.get_cycle_length(path1) + self.get_cycle_length(path2)
        vis.keep_graph()
        vis.update_graph(path1=path1, path2=path2, distance=path_len)
        return path1, path2, improv, percent

    def algo_test(self, filename):
        f = open(filename, "w")
        for x in range(100):
            p_len = self.regret(s=x)
            print("test", x, "len:", p_len)
            f.write(str(p_len) + '\n')
        f.close()

    def generate_random_paths(self):
        ran = [i for i in range(len(self.dist_matrix))]
        split = int(len(self.dist_matrix) / 2)
        a_random_path = random.sample(ran, split)
        a_random_path.append(a_random_path[0])
        b_random_path = [i for i in ran if i not in a_random_path]
        shuffle(b_random_path)
        b_random_path.append(b_random_path[0])
        #print("\n\nPATHS:", a_random_path, b_random_path)
        return a_random_path, b_random_path

    def construction_test(self, filename1, filename2, filename3):
        f1 = open(filename1, "w")
        f2 = open(filename2, "w")
        f3 = open(filename3, "w")
        best_before_1, best_before_2, best_after_1, best_after_2 = None, None, None, None
        best_percent = 0

        for x in range(100):
            #p1, p2 = self.generate_random_paths()
            p1, p2 = self.regret(x)
            start = time.time()
            print("building...")
            p1after, p2after, improv, percent = self.steepest(path1=p1, path2=p2, mode='e')
            end = time.time()
            elapsed = end - start
            print("test", x, "improvement:", improv, "percent", percent, "time:", elapsed)

            if percent > best_percent:
                best_before_1, best_before_2 = p1, p2
                best_after_1, best_after_2 = p1after, p2after
                best_percent = percent
                print("best", best_percent)

            f1.write(str(improv) + '\n')
            f2.write(str(elapsed) + '\n')
            f3.write(str(percent) + '\n')
        f1.close()
        f2.close()
        f3.close()
        return best_before_1, best_before_2, best_after_1, best_after_2

    def draw_path(self, p1, p2):
        cluster1 = [x for x in range(len(self.dist_matrix)) if x in p1]
        cluster2 = [x for x in range(len(self.dist_matrix)) if x in p2]
        vis = TSP_visualiser(self.coords, 'Best of steepest e regret, kroA100', cluster1, cluster2)
        vis.keep_graph()
        dist = self.get_cycle_length(p1) + self.get_cycle_length(p2)
        vis.update_graph(path1=p1, path2=p2, distance=dist)


solver = TSP_solver(kroA100_filename)
p1, p2 = solver.generate_random_paths()
N = solver.steepest(p1, path2=p2, mode='e')

# p1b, p2b, p1a, p2a = solver.construction_test("improv_test.txt", "time_test.txt", "percent_test.txt")
# print(p1b, p2b)
# solver.draw_path(p1a, p2a)