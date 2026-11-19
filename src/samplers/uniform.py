"""Uniform sampler for motion planning"""

from __future__ import annotations

import numpy as np

from core.state import State
from .base import BaseSampler, BaseValidStateSampler
from .space_info import SpaceInformation


class UniformSampler(BaseSampler):
    """
    Uniformly samples states across the bounded configuration space.
    """

    def sample_uniform(self, state: State):
        self._validate_state(state)

        state.values[:] = self.rng.uniform(self.space.low, self.space.high)

    def sample_uniform_near(self, state: State, near_state: State, distance: float):
        self._validate_state(state)
        self._validate_state(near_state)

        if distance < 0:
            raise ValueError("Distance must be non-negative")

        low = np.maximum(self.space.low, near_state.values - distance)
        high = np.minimum(self.space.high, near_state.values + distance)

        state.values[:] = self.rng.uniform(low, high)

    def sample_gaussian(self, state: State, mean: State, std: float):
        self._validate_state(state)
        self._validate_state(mean)

        if std < 0:
            raise ValueError("Standard deviation must be non-negative")

        state.values[:] = self.rng.normal(mean.values, std, size=self.space.dimension)

        self.space.enforce_bounds(state)


class UniformValidStateSampler(BaseValidStateSampler):
    """Uniform valid state sampler"""

    sampler_class = UniformSampler

    def __init__(self, space_info: SpaceInformation, max_attempts: int=100, seed = None):
        super().__init__(space_info, name="UniformValidStateSampler")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.max_attempts = max_attempts

        # Allocate an underlying state sampler.
        self.sampler = self.si.alloc_state_sampler(self.sampler_class, seed=seed)

    def sample(self, state):
        """
        Sample a uniformly distributed valid state.
        """

        self.sampler._validate_state(state)

        candidate = State(state.values.copy())
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform(candidate)
            if self.si.is_valid(candidate):
                state.values[:] = candidate.values
                return True

        return False

    def sample_near(self, state, near_state, distance = 0):
        """
        Sample a valid state within a neighborhood
        of near_state.
        """

        self.sampler._validate_state(state)
        self.sampler._validate_state(near_state)

        if distance < 0:
            raise ValueError("Distance must be non-negative.")

        candidate = State(state.values.copy())
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform_near(candidate, near_state, distance)
            if self.si.is_valid(candidate):
                state.values[:] = candidate.values
                return True

        return False