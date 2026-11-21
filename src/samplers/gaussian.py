"""Gaussian sampler (samples near obstacle boundaries)"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import BaseValidStateSampler
from .space_info import SpaceInformation
from .uniform import UniformSampler
from src.core.state import State


class GaussianSampler(UniformSampler):
    """
    Gaussian state sampler
    """
    pass


class GaussianValidStateSampler(BaseValidStateSampler):
    """Gaussian valid state sampler"""

    sampler_class = GaussianSampler
    STD_DEV_AS_SPACE_EXTENT_FRACTION = 0.1

    def __init__(
        self,
        space_info: SpaceInformation,
        max_attempts: int = 100,
        std: Optional[float] = None,
        seed = None
    ):
        super().__init__(space_info, name="GaussianValidStateSampler")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.max_attempts = max_attempts

        # Allocate an underlying state sampler.
        self.sampler = self.si.alloc_state_sampler(self.sampler_class, seed=seed)

        if not std:
            std = self.si.get_maximum_extent() * self.STD_DEV_AS_SPACE_EXTENT_FRACTION

        self.set_std(std)

    def set_std(self, std: float):
        if not np.isfinite(std) or std < 0:
            raise ValueError("Standard deviation must be finite and non-negative")

        self.std = float(std)

    def get_std(self):
        return self.std

    def set_max_attempts(self, attempts: int):
        if attempts <= 0:
            raise ValueError("Attempts must be positive.")

        self.max_attempts = int(attempts)

    def get_max_attempts(self):
        return self.max_attempts

    def sample(self, state):
        """
        Sample a uniformly distributed valid state.
        """

        self.sampler._validate_state(state)

        temp = State(np.empty_like(state.values))

        for _ in range(self.max_attempts):
            self.sampler.sample_uniform(state)
            valid1 = self.si.is_valid(state)

            self.sampler.sample_gaussian(temp, state, self.std_dev)
            valid2 = self.si.is_valid(temp)

            if valid1 != valid2:
                if valid2:
                    state.values[:] = temp.values
                return True

        return False

    def sample_near(self, state, near_state, distance = 0):
        """
        Sample a valid state within a neighborhood
        of near_state.
        """

        self.sampler._validate_state(state)
        self.sampler._validate_state(near_state)

        if distance < 0 or not np.isfinite(distance):
            raise ValueError("Distance must be finite and non-negative")

        temp = State(np.empty_like(state.values))
        for _ in range(self.max_attempts):
            # Generate first candidate near reference.
            self.sampler.sample_uniform_near(state, near_state, distance)
            valid1 = self.si.is_valid(state)

            # Generate second candidate around first.
            self.sampler.sample_gaussian(temp, state, distance)
            valid2 = self.si.is_valid(temp)
            if valid1 != valid2:
                if valid2:
                    state.values[:] = temp.values
                return True

        return False