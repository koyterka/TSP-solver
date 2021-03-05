import matplotlib.pyplot as plt
from matplotlib.pyplot import ion

class TSP_visualiser:
    def __init__(self, coords, title, cluster1, cluster2):
        self.color1 = '#ff0e8e'
        self.color1_1 = '#ff86c6'
        self.color2 = '#5706ad'
        self.color2_2 = '#ab82d6'
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.coords = coords
        self.title = title
        self.set_stuff()
        self.edges = []
        self.path1 = None
        self.path2 = None
        ion()
        plt.show()

    def set_stuff(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self.title)
        cluster1_coords = [self.coords[x] for x in self.cluster1]
        cluster2_coords = [self.coords[x] for x in self.cluster2]
        self.ax.scatter(cluster1_coords[0][0], cluster1_coords[0][1], c=self.color1)
        self.ax.scatter([p[0] for p in cluster1_coords[1:]], [p[1] for p in cluster1_coords[1:]], c=self.color1_1)
        self.ax.scatter(cluster2_coords[0][0], cluster2_coords[0][1], c=self.color2)
        self.ax.scatter([p[0] for p in cluster2_coords[1:]], [p[1] for p in cluster2_coords[1:]], c=self.color2_2)

    def draw_edge(self, a, b, color):
        edge = self.ax.annotate("",
                    xy=a, xycoords='data',
                    xytext=b, textcoords='data',
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3", color=color))
        self.edges.append(edge)

    def draw_path(self, path):
        if path:
            color = self.color1_1
            if path[0] == self.cluster2[0]:
                color = self.color2_2
            for x in range(len(path) - 1):
                start_pos = self.coords[path[x]]
                end_pos = self.coords[path[x + 1]]
                self.draw_edge(start_pos, end_pos, color)
            self.draw_edge(self.coords[path[-1]], self.coords[path[0]], color)

    def update_graph(self, path1=None, path2=None, distance=0):
        for i, a in enumerate(self.edges):
            a.remove()
        self.edges[:] = []
        #self.set_stuff()
        if path1: self.path1 = path1
        if path2: self.path2 = path2

        self.draw_path(self.path1)
        self.draw_path(self.path2)

        # show the graph
        if distance!=0:
            distance = distance
            N = len(self.coords)
            textstr = "N nodes: %d\nTotal length: %d" % (N, distance)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        plt.tight_layout()
        #ion()
        plt.show()
        plt.pause(1e-20)

    def keep_graph(self):
        plt.ioff()

