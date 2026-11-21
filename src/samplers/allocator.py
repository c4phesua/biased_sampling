"""Valid state sampler allocator"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from samplers.space_info import SpaceInformation
    from samplers.base import BaseValidStateSampler


class ValidStateSamplerAllocator:
    """
    Factory for allocating valid state samplers
    """

    def __init__(self, allocator: Optional[Callable] = None):
        self._allocator = None
        if allocator is not None:
            self.set_allocator(allocator)

    def set_allocator(self, allocator: Callable):
        if not callable(allocator):
            raise TypeError("Allocator must be callable")

        self._allocator = allocator

    def clear(self):
        self._allocator = None

    def is_set(self) -> bool:
        return self._allocator is not None

    def allocate(
        self,
        space_information: "SpaceInformation"
    ) -> "BaseValidStateSampler":

        if self._allocator is None:
            raise RuntimeError("No valid state sampler allocator registered")

        sampler = self._allocator(space_information)

        from src.samplers.base import BaseValidStateSampler

        if not isinstance(sampler, BaseValidStateSampler):
            raise TypeError("Allocator must return a BaseValidStateSampler instance")

        return sampler

    def __call__(
        self,
        space_information: "SpaceInformation"
    ) -> "BaseValidStateSampler":

        return self.allocate(space_information)