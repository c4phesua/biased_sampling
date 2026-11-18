"""Configuration domain"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .state import State


class BaseStateSpace(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def sample_uniform(self, rng) -> State:
        pass

    @abstractmethod
    def satisfies_bounds(self, state: State) -> bool:
        pass

    @abstractmethod
    def enforce_bounds(self, state: State) -> None:
        pass

    @abstractmethod
    def distance(self, state1: State, state2: State) -> float:
        pass

    @abstractmethod
    def get_maximum_extent(self) -> float:
        pass

    @abstractmethod
    def interpolate(self, state1, state2, t, state):
        pass


class RealVectorStateSpace(BaseStateSpace):

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        self._dimension = dimension
        self.low = np.full(dimension, -1.0)
        self.high = np.full(dimension, 1.0)

    @property
    def dimension(self) -> int:
        return self._dimension

    def set_bounds(self, low, high):
        low = np.broadcast_to(
            np.asarray(low, dtype=np.float64), (self.dimension,)
        ).copy()

        high = np.broadcast_to(
            np.asarray(high, dtype=np.float64), (self.dimension,)
        ).copy()

        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Bounds must be finite.")

        if np.any(low > high):
            raise ValueError("Lower bounds must not exceed upper bounds.")

        self.low = low
        self.high = high

    def sample_uniform(self, rng) -> State:
        values = rng.uniform(self.low, self.high)
        return State(values)

    def satisfies_bounds(self, state: State) -> bool:
        self._check_state(state)

        return bool(
            np.all(state.values >= self.low)
            and np.all(state.values <= self.high)
        )

    def enforce_bounds(self, state: State) -> None:
        self._check_state(state)
        state.values[:] = np.clip(
            state.values, self.low, self.high
        )

    def distance(self, state1: State, state2: State) -> float:
        self._check_state(state1)
        self._check_state(state2)

        return float(np.linalg.norm(state1.values - state2.values))

    def get_maximum_extent(self) -> float:
        return float(np.linalg.norm(self.high - self.low))

    def interpolate(self, state1: State, state2: State, t: float, state: State):
        """
        Linear interpolation between two states
        """
        self._check_state(state1)
        self._check_state(state2)
        self._check_state(state)

        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be in the range between 0 and 1")

        state.values[:] = (1.0 - t) * state1.values + t * state2.values

    def _check_state(self, state: State):
        if state.dimension != self.dimension:
            raise ValueError("State dimension does not match state space.")