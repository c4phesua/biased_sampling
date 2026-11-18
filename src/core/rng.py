"""Random number generator for sampling"""

from __future__ import annotations

import numpy as np


class RandomNumberGenerator:
    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def uniform(self, low, high):
        return self._rng.uniform(low, high)

    def uniform_int(self, low, high):
        return int(self._rng.integers(low, high+1))

    def gaussian(self, mean=0.0, std=0.0):
        return self._rng.normal(mean, std)

    def normal(self, mean, std, size=None):
        return self._rng.normal(mean, std, size)