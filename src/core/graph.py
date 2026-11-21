"""Graph using State as nodes"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from src.core.state import State


class NodeGraph:
    def __init__(self):
        self.V = []
        self.kdtree = None  # for run k nearest neighbors
        self.nx_graph = nx.Graph()  # for find the shortest distance A*
        self._state_by_key = {}

    @staticmethod
    def _key(state):
        if not isinstance(state, State):
            raise TypeError("Expected a State instance.")

        return tuple(float(value) for value in state.values)

    def add_node(self, state: State):
        key = self._key(state)
        if key in self._state_by_key:
            return False
        
        state_clone = state.copy()
        self.V.append(state_clone)
        self._state_by_key[key] = state_clone
        self.nx_graph.add_node(key)
        self.kdtree = KDTree(np.asarray([node.values for node in self.V]))
        return True

    def add_edge(self, u: State, v: State):
        key_u = self._key(u)
        key_v = self._key(v)
        if key_u not in self.nx_graph or key_v not in self.nx_graph:
            raise ValueError("Both states must be in the graph.")
        if key_u == key_v:
            return

        dist = u.distance(v)
        self.nx_graph.add_edge(key_u, key_v, weight=dist)

    def k_nn_search(self, q_rand: State, k):
        if k <= 0:
            raise ValueError("k must be positive")

        if not self.V:
            return []

        query = np.asarray(q_rand.values, dtype=np.float64)
        k_actual = min(k + 1, len(self.V))
        _, indices = self.kdtree.query(query, k=k_actual)
        indices = np.atleast_1d(indices)

        query_key = self._key(q_rand)
        neighbors = []
        for index in indices:
            neighbor = self.V[int(index)]
            if self._key(neighbor) == query_key:
                continue

            neighbors.append(neighbor)
            if len(neighbors) == k:
                break

        return neighbors

    @staticmethod
    def cal_euclid_distance(key1, key2):
        return math.dist(key1, key2)

    def get_shortest_path(self, q_start, q_goal):
        start_key = self._key(q_start)
        goal_key = self._key(q_goal)
        if start_key not in self.nx_graph or goal_key not in self.nx_graph:
            return None
        try:
            path = nx.astar_path(
                self.nx_graph,
                source=start_key,
                target=goal_key,
                heuristic=self.cal_euclid_distance,
                weight='weight'
            )
            return [
                self._state_by_key[key].copy()
                for key in path
            ]
        except nx.NetworkXNoPath:
            return None
