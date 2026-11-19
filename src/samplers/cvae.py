"""Sampler using Conditional Variational Autoencoder, per Ichter et al. 2018"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import BaseValidStateSampler
from .space_info import SpaceInformation
from .uniform import UniformSampler
from core.state import State
from core.state_space import RealVectorStateSpace
from core.rng import RandomNumberGenerator
from models.cvae import CVAE


class CVAESampler(UniformSampler):
    """
    State sampler using CVAE
    """
    def __init__(
        self,
        space: RealVectorStateSpace,
        cvae: Optional[CVAE],
        condition = None,
        seed = None
    ):
        super().__init__(space, seed)
        if cvae is None:
            raise ValueError("CVAE model should be input")

        if cvae.state_dim != space.dimension:
            raise ValueError("CVAE state dimension does not match space dimension")

        self.cvae = cvae
        self.condition = condition

        self._torch_generator = None

        if seed is not None:
            self._torch_generator = torch.Generator(
                device=self.cvae.device
            )
            self._torch_generator.manual_seed(seed)

    def set_condition(self, condition):
        """Set the planning-problem conditioning vector."""

        self.condition = condition

    def sample_cvae(self, state, condition=None):
        """
        Generate a state using the CVAE decoder.
        """

        self._validate_state(state)

        if condition is None:
            condition = self.condition

        samples = self.cvae.generate(
            y=condition,
            n_samples=1,
            generator=self._torch_generator,
        )

        state.values[:] = samples[0]

    def sample_learned(self, state, condition=None):
        """Alias for sample_cvae()"""

        self.sample_cvae(state, condition)


class CVAEValidStateSampler(BaseValidStateSampler):
    """
    Valid state sampler using CVAE architecture
    """

    sampler_class = CVAESampler

    def __init__(
        self,
        space_info: SpaceInformation,
        cvae: Optional[CVAE],
        condition = None,
        learned_fraction: float = 0.5,
        max_attempts: int = 100,
        seed = None
    ):
        super().__init__(space_info, name="CVAEValidStateSampler")

        if not 0.0 <= learned_fraction <= 1.0:
            raise ValueError("learned_fraction must be between 0 and 1")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.learned_fraction = float(learned_fraction)
        self.max_attempts = max_attempts

        # Allocate an underlying state sampler.
        self.sampler = CVAESampler(
            space=self.si.space,
            cvae=cvae,
            condition=condition,
            seed=seed,
        )

        self.rng = RandomNumberGenerator(seed)

    def set_condition(self, condition):
        """Update the condition for the current planning problem."""

        self.sampler.set_condition(condition)

    def set_learned_fraction(self, fraction):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")

        self.learned_fraction = float(fraction)

    def get_learned_fraction(self):
        return self.learned_fraction

    def set_max_attempts(self, attempts: int):
        if attempts <= 0:
            raise ValueError("Attempts must be positive.")

        self.max_attempts = int(attempts)

    def get_max_attempts(self):
        return self.max_attempts

    def _sample_candidate(self, candidate: State):
        """
        Select one branch and generate a candidate
        """
        uses_cvae = self.rng.uniform01() < self.learned_fraction
        if uses_cvae:
            self.sampler.sample_cvae(candidate)
        else:
            self.sampler.sample_uniform(candidate)

    def _sample_candidate_near(self, candidate: State, near_state: State, distance: float):
        uses_cvae = self.rng.uniform01() < self.learned_fraction
        if uses_cvae:
            self.sampler.sample_cvae(candidate)
        else:
            self.sampler.sample_uniform_near(candidate, near_state, distance)
        
        return self.si.space.distance(candidate, near_state) > distance

    def sample(self, state):
        self.sampler._validate_state(state)
        candidate = state.copy()
        for _ in range(self.max_attempts):
            self._sample_candidate(candidate)
            if self.si.is_valid(candidate):
                state.values[:] = candidate.values
                return True
        return False

    def sample_near(self, state: State, near_state: State, distance: float = 0):
        self.sampler._validate_state(state)
        self.sampler._validate_state(near_state)

        if distance < 0 or not np.isfinite(distance):
            raise ValueError("Distance must be finite and non-negative")

        candidate = state.copy()
        for _ in range(self.max_attempts):
            if self._sample_candidate_near(candidate, near_state, distance):
                continue
            if self.si.is_valid(candidate):
                state.values[:] = candidate.values
                return True

        return False