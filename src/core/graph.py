import math

import networkx as nx
from scipy.spatial import cKDTree


class NodeGraph:
    def __init__(self):
        self.V = []
        self.kdtree = None  # for run k nearest neighbors
        self.nx_graph = nx.Graph()  # for find the shortest distance A*

    def add_node(self, q_rand):
        self.V.append(q_rand)
        self.kdtree = cKDTree(self.V)
        self.nx_graph.add_node(q_rand)

    def add_edge(self, u, v):
        dist = math.dist(u, v)
        self.nx_graph.add_edge(u, v, weight=dist)

    def k_nn_search(self, q_rand, k):
        k_actual = min(k, len(self.V))
        if k_actual == 0:
            return []
        distances, indices = self.kdtree.query(q_rand, k=k_actual)
        if k_actual == 1:
            indices = [indices]
        neighbors = [self.V[i] for i in indices]
        return neighbors

    @staticmethod
    def cal_euclid_distance(u, v):
        return math.dist(u, v)

    def get_shortest_path(self, q_start, q_goal):
        if q_start not in self.nx_graph or q_goal not in self.nx_graph:
            return None
        try:
            path = nx.astar_path(
                self.nx_graph,
                source=q_start,
                target=q_goal,
                heuristic=self.cal_euclid_distance,
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            return None
