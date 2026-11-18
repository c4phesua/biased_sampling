"""Conditional Variational Autoencoder, per Ichter et al. 2018"""

from __future__ import annotations

from .base import BaseValidStateSampler
from .space_info import SpaceInformation
from .uniform import UniformSampler
from core.state import State


class CVAESampler(UniformSampler):
    """
    State sampler using CVAE
    """
    pass


class CVAEValidStateSampler(BaseValidStateSampler):
    """
    Valid state sampler using CVAE architecture
    """

    sampler_class = CVAESampler

    def __init__(
        self,
        space_info: SpaceInformation,
        max_attempts: int = 100,
        seed = None
    ):
        super().__init__(space_info, name="CVAEValidStateSampler")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.max_attempts = max_attempts

        # Allocate an underlying state sampler.
        self.sampler = self.si.alloc_state_sampler(self.sampler_class, seed=seed)