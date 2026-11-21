"""Obstacle based sampler (OBPRM)"""

from __future__ import annotations

from .base import BaseValidStateSampler
from .space_info import SpaceInformation
from .uniform import UniformSampler
from src.core.state import State


class ObstacleBasedSampler(UniformSampler):
    """
    Obstacle-based state sampler
    """
    pass


class ObstacleBasedValidStateSampler(BaseValidStateSampler):
    """
    Obstacled-based valid state sampler
    """

    sampler_class = ObstacleBasedSampler

    def __init__(
        self,
        space_info: SpaceInformation,
        max_attempts: int = 100,
        seed = None
    ):
        super().__init__(space_info, name="ObstacleBasedValidStateSampler")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.max_attempts = max_attempts

        # Allocate an underlying state sampler.
        self.sampler = self.si.alloc_state_sampler(self.sampler_class, seed=seed)

    def set_max_attempts(self, attempts: int):
        if attempts <= 0:
            raise ValueError("Attempts must be positive.")

        self.max_attempts = int(attempts)

    def get_max_attempts(self):
        return self.max_attempts
    
    def sample(self, state):
        self.sampler._validate_state(state)

        # Find an invalid state
        has_invalid = False
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform(state)
            if not self.si.is_valid(state):
                has_invalid = True
                break
        if not has_invalid:
            return False

        # Find a valid state
        temp = State(state.values.copy())
        has_valid = False
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform(temp)
            if self.si.is_valid(temp):
                has_valid = True
                break
        if not has_valid:
            return False

        # Find the last valid state along the motion from temp to state
        self.si.check_motion(temp, state, last_valid_state=state)

        return True

    def sample_near(self, state, near_state, distance = 0):
        self.sampler._validate_state(state)
        self.sampler._validate_state(near_state)

        if distance < 0:
            raise ValueError("Distance must be non-negative")

        # Find an invalid state near near_state
        has_invalid = False

        for _ in range(self.max_attempts):
            self.sampler.sample_uniform_near(state, near_state, distance)
            if not self.si.is_valid(state):
                has_invalid = True
                break
        if not has_invalid:
            return False

        # Find a valid state
        temp = State(state.values.copy())
        has_valid = False
        for _ in range(self.max_attempts):
            self.sampler.sample_uniform(temp)
            if self.si.is_valid(temp):
                has_valid = True
                break
        if not has_valid:
            return False

        # Find the last valid state.
        self.si.check_motion(temp, state, last_valid_state=state)

        return True