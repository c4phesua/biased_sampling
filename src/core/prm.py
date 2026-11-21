"""Probabilistic Roadmap Planner"""

from __future__ import annotations

from src.core.graph import NodeGraph
from src.core.state import State


class PRM2D:
    def __init__(
        self,
        space_information,
        graph: NodeGraph,
        q_start: State,
        q_goal: State,
        sampler=None,
        motion_resolution=0.005
    ):
        if space_information is None:
            raise ValueError("SpaceInformation cannot be null.")

        if graph is None:
            raise ValueError("Graph cannot be null.")

        self.si = space_information
        self.graph = graph
        self.q_start = q_start.copy()
        self.q_goal = q_goal.copy()

        if self.q_start.dimension != 2 or self.q_goal.dimension != 2:
            raise ValueError("PRM2D requires 2D states.")

        if not self.si.is_valid(self.q_start):
            raise ValueError("Start state is invalid.")

        if not self.si.is_valid(self.q_goal):
            raise ValueError("Goal state is invalid.")

        if motion_resolution <= 0:
            raise ValueError("Motion resolution must be positive.")

        self.motion_resolution = float(motion_resolution)

        self.sampler = (
            sampler
            if sampler is not None
            else self.si.alloc_valid_state_sampler()
        )

        self.graph.add_node(self.q_start)
        self.graph.add_node(self.q_goal)

    def _sample_valid_state(self):
        """Sample one valid state using the configured sampler."""
        state = State([0.0, 0.0])

        if self.sampler.sample(state):
            return state

        return None

    def _connect_neighbors(self, state, neighbors):
        for neighbor in neighbors:
            last_valid = neighbor.copy()

            if self.si.check_motion(
                neighbor,
                state,
                last_valid_state=last_valid,
                resolution=self.motion_resolution
            ):
                self.graph.add_edge(state, neighbor)

    def run(self, sample_num, k: int = 5, max_attempts=None):
        if sample_num < 0:
            raise ValueError("Sample numbers must be non-negative")
        if k <= 0:
            raise ValueError("k must be positive")
        if max_attempts is None:
            max_attempts = max(100, sample_num * 50)
        
        added = 0
        attempts = 0
        while added < sample_num and attempts < max_attempts:
            attempts += 1
            q_rand = self._sample_valid_state()
            if q_rand is None:
                continue

            neighbors = self.graph.k_nn_search(q_rand, k)
            if not self.graph.add_node(q_rand):
                continue

            self._connect_neighbors(q_rand, neighbors)
            added += 1

        return self.graph.get_shortest_path(self.q_start, self.q_goal)
