"""Space Information"""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.core.state import State
from src.core.state_space import BaseStateSpace
from .allocator import ValidStateSamplerAllocator


class SpaceInformation:
    def __init__(self, space: BaseStateSpace, check_validity_fn, check_motion_fn=None):
        if space is None:
            raise ValueError("State space cannot be null.")
        
        if not callable(check_validity_fn):
            raise TypeError("Validity checker must be callable.")

        self.space = space
        self.check_validity_fn = check_validity_fn

        self._valid_state_sampler_allocator = (
            ValidStateSamplerAllocator()
        )

        if check_motion_fn is not None and not callable(check_motion_fn):
            raise TypeError("Motion validity checker must be callable.")
        self.check_motion_fn = check_motion_fn

    def set_valid_state_sampler_allocator(self, allocator):
        """
        Register a custom valid-state sampler allocator
        """

        if isinstance(allocator, ValidStateSamplerAllocator):
            if not allocator.is_set():
                raise ValueError("Allocator has no registered factory.")

            self._valid_state_sampler_allocator = allocator
        elif callable(allocator):
            self._valid_state_sampler_allocator = ValidStateSamplerAllocator(allocator)
        else:
            raise TypeError("Expected ValidStateSamplerAllocator or callable.")

    def clear_valid_state_sampler_allocator(self):
        """
        Restore the default UniformValidStateSampler
        """
        self._valid_state_sampler_allocator.clear()

    def alloc_valid_state_sampler(self):
        """
        Allocate a valid-state sampler

        Returns:
            Custom sampler if an allocator is registered.
            UniformValidStateSampler otherwise.
        """

        if self._valid_state_sampler_allocator.is_set():
            return self._valid_state_sampler_allocator.allocate(self)

        # Lazy import to avoid circular dependencies.
        from src.samplers.uniform import UniformValidStateSampler

        return UniformValidStateSampler(self)

    def is_valid(self, state: State):
        if not self.space.satisfies_bounds(state):
            return False

        return bool(self.check_validity_fn(state))

    def alloc_state_sampler(self, sampler_class, seed=None):
        return sampler_class(self.space, seed=seed)

    def get_maximum_extent(self) -> float:
        return self.space.get_maximum_extent()

    def interpolate(self, state1: State, state2: State, t: float, state: State):
        return self.space.interpolate(state1, state2, t, state)

    def check_motion(
        self,
        state1: State,
        state2: State,
        last_valid_state: Optional[State] = None,
        resolution=0.01
    ):
        """
        Check motion validity between `state1` and `state2`

        If `last_valid_state` is supplied, it is populated 
        with the last valid configuration encountered.
        """
        if not self.is_valid(state1):
            raise ValueError("Starting state must be valid")

        if resolution <= 0:
            raise ValueError("Resolution must be positive")

        if self.check_motion_fn is not None:
            if not self.check_motion_fn(state1, state2):
                if last_valid_state is not None:
                    last_valid_state.values[:] = state1.values
                return False

        distance = self.space.distance(state1, state2)
        extent = self.get_maximum_extent()
        segments = max(1, int(np.ceil(distance / (resolution * extent)))) if extent > 0 else 1

        previous = state1.copy()
        current = State(state1.values.copy())
        for i in range(1, segments + 1):
            t = i / segments
            self.interpolate(state1, state2, t, current)
            if not self.is_valid(current):
                if last_valid_state is not None:
                    last_valid_state.values[:] = previous.values
                return False
            previous.values[:] = current.values

        if last_valid_state is not None:
            last_valid_state.values[:] = state2.values

        return True