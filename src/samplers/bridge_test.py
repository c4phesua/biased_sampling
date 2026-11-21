"""Bridge test sampler (designed for narrow passages)"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import BaseValidStateSampler
from .space_info import SpaceInformation
from .uniform import UniformSampler
from src.core.state import State


class BridgeTestSampler(UniformSampler):
    """
    Bridge test state sampler
    """
    pass


class BridgeTestValidStateSampler(BaseValidStateSampler):
    """Bridge test valid state sampler"""

    sampler_class = BridgeTestSampler
    STD_DEV_AS_SPACE_EXTENT_FRACTION = 0.1

    def __init__(
        self,
        space_info: SpaceInformation,
        max_attempts: int = 100,
        std: Optional[float] = None,
        seed = None
    ):
        super().__init__(space_info, name="BridgeTestValidStateSampler")

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
        self.sampler._validate_state(state)

        endpoint = State(state.values.copy())
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform(state)
            if self._bridge_test(state, endpoint, self.std):
                return True

        return False

    def sample_near(self, state, near_state, distance = 0):
        self.sampler._validate_state(state)
        self.sampler._validate_state(near_state)

        if distance < 0 or not np.isfinite(distance):
            raise ValueError("Distance must be finite and non-negative")

        endpoint = State(state.values.copy())
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform_near(state, near_state, distance)
            if self._bridge_test(state, endpoint, distance):
                return True

        return False

    def _bridge_test(self, state: State, endpoint: State, std: float):
        """
        Perform bridge test attempt
        """
        # First endpoint must be invalid
        if self.si.is_valid(state):
            return False

        # Generate second endpoint
        self.sampler.sample_gaussian(endpoint, state, std)
        if self.si.is_valid(endpoint):
            return False

        self.si.interpolate(state, endpoint, 0.5, state)

        return self.si.is_valid(state)